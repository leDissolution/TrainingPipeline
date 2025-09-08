"""
Shared training utilities for TRL SFT scripts.

This module provides a flexible way to define optimizer parameter groups
by specifying which modules (via suffix substrings) and which layers to
target, with per-group learning rate and weight decay.

Key components:
- GroupSpec: dataclass describing a group (suffixes, layers, lr/scale, wd)
- FlexibleLRTrainer: SFTTrainer subclass that materializes groups from specs,
  logs groups like stage0, and warns for disabled params targeted by a group.

Note: Existing training scripts are untouched; you can adopt this trainer
progressively without removing current code in scripts.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, cast, Sized
from dataclasses import dataclass
import os
import json

import torch
from trl import SFTTrainer
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DistributedSampler, IterableDataset

# ---------------------------------------------------------------------------
# Group specification (order matters; first match wins)
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    """Specification for a parameter group.

    - name: identifier for the group (informational)
    - suffixes: list of substrings to match (e.g., ["q_proj"], ["mlp.up_proj"], ["embed_tokens"])
    - layers: optional list/range of layer indices to restrict (requires name to contain f"layers.{i}.")
    - lr: optional absolute LR for this group; if None, computed from base_lr * lr_scale
    - lr_scale: optional multiplicative factor applied to base_lr when lr is None
    - weight_decay: optional per-group weight decay (overrides optimizer default)
    """

    name: str
    suffixes: Sequence[str]
    layers: Optional[Sequence[int]] = None
    lr: Optional[float] = None
    lr_scale: Optional[float] = None
    weight_decay: Optional[float] = None
    # Optional one-time weight rescale applied to matched parameters before training
    # Useful for nudging specific layers/modules (e.g., mlp up/down/gate) without code edits
    weight_rescale: Optional[float] = None


def _layer_token(i: int) -> str:
    return f"layers.{i}."


def _name_matches_spec(param_name: str, spec: GroupSpec) -> bool:
    if spec.layers is not None:
        if not any(_layer_token(i) in param_name for i in spec.layers):
            return False
    # suffix match: any of the provided substrings
    return any(sfx in param_name for sfx in spec.suffixes)


def build_param_groups_from_specs(
    model: torch.nn.Module,
    specs: Sequence[GroupSpec],
    base_lr: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Create optimizer param_groups based on ordered GroupSpec list.

    Returns (param_groups, disabled_counts_by_group).

    Any parameter that does not match any spec goes into the base group
    at base_lr. Only includes requires_grad parameters.

    Special handling:
    - If a spec resolves to an effective learning rate of 0 (either lr == 0
      or lr_scale == 0), all parameters matched by that spec are frozen by
      setting requires_grad=False and are not added to any optimizer group.
      This avoids confusing the optimizer with zero-lr groups and makes the
      freeze intent explicit.
    """
    matched_sets: List[List[torch.nn.Parameter]] = [[] for _ in specs]
    disabled_counts: Dict[str, int] = {spec.name: 0 for spec in specs}
    base: List[torch.nn.Parameter] = []

    def _effective_lr(spec: GroupSpec) -> float:
        if spec.lr is not None:
            return float(spec.lr)
        scale = 1.0 if spec.lr_scale is None else float(spec.lr_scale)
        return float(base_lr * scale)

    # Determine which specs imply freezing (effective lr == 0)
    freeze_spec: List[bool] = [(_effective_lr(s) == 0.0) for s in specs]

    for n, p in model.named_parameters():
        placed = False
        for i, spec in enumerate(specs):
            if _name_matches_spec(n, spec):
                if freeze_spec[i]:
                    # Freeze matched params for zero-lr groups
                    if p.requires_grad:
                        p.requires_grad = False
                    disabled_counts[spec.name] = disabled_counts.get(spec.name, 0) + 1
                else:
                    if p.requires_grad:
                        matched_sets[i].append(p)
                    else:
                        disabled_counts[spec.name] = disabled_counts.get(spec.name, 0) + 1
                placed = True
                break  # first match wins
        if not placed and p.requires_grad:
            base.append(p)

    param_groups: List[Dict[str, Any]] = []
    # Add explicit groups
    for spec, params in zip(specs, matched_sets):
        lr = spec.lr if spec.lr is not None else base_lr * (spec.lr_scale if spec.lr_scale is not None else 1.0)
        group: Dict[str, Any] = {"params": params, "lr": float(lr)}
        if spec.weight_decay is not None:
            group["weight_decay"] = float(spec.weight_decay)
        param_groups.append(group)

    # Add base group last
    param_groups.append({"params": base, "lr": float(base_lr)})
    return param_groups, disabled_counts


class LayerwiseLRTrainer(SFTTrainer):
    """Generic trainer that accepts arbitrary group specifications.

    Example:
        specs = [
            GroupSpec(name="q", suffixes=["q_proj"], layers=range(10, 20), lr_scale=1.0),
            GroupSpec(name="k", suffixes=["k_proj"], layers=range(10, 20), lr_scale=1.0),
            GroupSpec(name="o", suffixes=["o_proj"], layers=range(10, 20), lr_scale=0.8, weight_decay=0.0),
            GroupSpec(name="up", suffixes=["mlp.up_proj"], layers=range(10, 20), lr_scale=0.1),
            GroupSpec(name="gate", suffixes=["mlp.gate_proj"], layers=range(10, 20), lr_scale=0.2),
            GroupSpec(name="embed", suffixes=["embed_tokens", "lm_head"], lr_scale=0.01),
        ]
        trainer = FlexibleLRTrainer(..., group_specs=specs, base_lr=3e-5)
    """

    def __init__(
        self,
        *args,
        group_specs: Optional[Sequence[GroupSpec]] = None,
        base_lr: float = 1e-4,
        dataset_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_lr = float(base_lr)
        self._specs: List[GroupSpec] = list(group_specs) if group_specs is not None else []
        # Optional separate RNG seed for dataset shuffling, independent of self.args.seed
        self._dataset_seed: Optional[int] = int(dataset_seed) if dataset_seed is not None else None
        self._dataset_generator: Optional[torch.Generator] = None
        if self._dataset_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self._dataset_seed)
            self._dataset_generator = gen

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is None:
            mdl = cast(torch.nn.Module, self.model)
            param_groups, disabled_counts = build_param_groups_from_specs(mdl, self._specs, base_lr=self._base_lr)
            for g in param_groups:
                g.setdefault("initial_lr", g["lr"])  # for schedulers

            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )

            # Log groups like stage0
            for i, g in enumerate(self.optimizer.param_groups):
                n_params = sum(p.numel() for p in g["params"]) if isinstance(g.get("params"), list) else 0
                print(i, "lr=", g.get("lr"), "wd=", g.get("weight_decay", self.args.weight_decay), "n_params=", n_params)

            # Warn about disabled targets
            for spec in self._specs:
                count = disabled_counts.get(spec.name, 0)
                if count and count > 0:
                    print(f"[Warning] Group '{spec.name}' matched {count} parameters with requires_grad=False. They were not included in optimization.")
        return self.optimizer

    def _get_train_sampler(self, idk = None):  # type: ignore[override]
        if self.train_dataset is None:
            return None
        if isinstance(self.train_dataset, IterableDataset):
            return None

        world_size = getattr(self.args, "world_size", 1)
        process_index = getattr(self.args, "process_index", 0)

        if world_size is None or world_size <= 1:
            ds_sized = cast(Sized, self.train_dataset)
            if self._dataset_generator is not None:
                return RandomSampler(ds_sized, generator=self._dataset_generator)
            return RandomSampler(ds_sized)
        else:
            seed = self._dataset_seed if self._dataset_seed is not None else int(getattr(self.args, "seed", 0))
            ds_any = cast(Any, self.train_dataset)
            return DistributedSampler(
                ds_any,
                num_replicas=world_size,
                rank=process_index,
                shuffle=True,
                seed=int(seed),
                drop_last=False,
            )

__all__ = [
    "GroupSpec",
    "build_param_groups_from_specs",
    "LayerwiseLRTrainer",
    # helpers
    "build_layered_suffixes",
    "build_target_modules",
    "apply_weight_rescale",
    # custom trainers
    "ForkWeighedLossTrainer",
]


# ---------------------------------------------------------------------------
# Helper functions to build layered suffix strings and LoRA target_modules
# ---------------------------------------------------------------------------

def build_layered_suffixes(
    layers: Iterable[int],
    module_suffix: str,
    layer_prefix: str = "layers",
) -> List[str]:
    return [f"{layer_prefix}.{int(i)}.{module_suffix}" for i in layers]


def build_target_modules(
    include_attention: bool = False,
    include_mlp: bool = False,
    include_modules: Iterable[str] = (),
    layered: Sequence[Tuple[Iterable[int], str]] = (),
    layer_prefix: str = "layers",
) -> List[str]:
    tm: List[str] = []
    if include_attention:
        tm.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if include_mlp:
        tm.extend(["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"])
    for layers, suffix in layered:
        tm.extend(build_layered_suffixes(layers, suffix, layer_prefix))

    tm.extend(include_modules)

    return tm


# ---------------------------------------------------------------------------
# One-time weight rescale based on GroupSpec matching (first-match-wins)
# ---------------------------------------------------------------------------

def apply_weight_rescale(model: torch.nn.Module, specs: Sequence[GroupSpec]) -> None:
    """Multiply parameter tensors by a per-group factor once, if provided.

    Semantics:
        - First matching spec wins (consistent with optimizer grouping).
        - Only applies when spec.weight_rescale is not None and != 1.0.
        - Applies to base weights only (name endswith ".weight"), and skips LoRA adapter
            parameters (names containing "lora_"), so it is safe to call before or after PEFT.
    """
    if not specs:
        return

    # Precompute which specs actually request a rescale
    active_specs: List[Tuple[int, GroupSpec]] = [
        (i, s) for i, s in enumerate(specs)
        if s.weight_rescale is not None and float(s.weight_rescale) != 1.0
    ]
    if not active_specs:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Only scale the base weights, not biases or lora adapter params
            if not name.endswith(".weight"):
                continue
            if "lora_" in name:
                continue
            # first-match-wins
            for _, spec in active_specs:
                if _name_matches_spec(name, spec):
                    try:
                        factor = float(spec.weight_rescale)  # type: ignore[arg-type]
                        param.mul_(factor)
                    except Exception:
                        pass
                    break


# ---------------------------------------------------------------------------
# Custom trainer: per-token loss weighting via example mask ("fork")
# ---------------------------------------------------------------------------

class ForkWeighedLossTrainer(LayerwiseLRTrainer):
    def __init__(
        self,
        *args,
        fork_alpha: float = 1.0,
        fork_mask_key: str = "fork_mask",
        ignore_index: int = -100,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fork_alpha = float(fork_alpha)
        self._fork_mask_key = str(fork_mask_key)
        self._ignore_index = int(ignore_index)
        # Per-example logging controls (opt-in via env vars)
        self._pe_log_topk: int = 0
        self._pe_log_text: bool = False
        self._pe_log_min_loss: float = 0.0

        try:
            self._pe_log_topk = int(os.environ.get("PER_EX_TRAIN_LOG_TOPK", "0"))
        except Exception:
            self._pe_log_topk = 0
        val = os.environ.get("PER_EX_TRAIN_LOG_TEXT", "0").strip().lower()
        self._pe_log_text = val in ("1", "true", "yes", "y")
        try:
            self._pe_log_min_loss = float(os.environ.get("PER_EX_TRAIN_LOG_MIN_LOSS", "0"))
        except Exception:
            self._pe_log_min_loss = 0.0
        self._pe_log_dir: Optional[str] = None
        self._pe_log_path: Optional[str] = None
        # scratch holder for last batch details in case callbacks want it
        self._last_batch_per_example: Optional[Dict[str, Any]] = None

    def _ensure_pe_log_path(self) -> Optional[str]:
        if self._pe_log_topk <= 0:
            return None
        # Only main process writes
        proc_idx = int(getattr(self.args, "process_index", 0) or 0)
        if proc_idx != 0:
            return None
        if self._pe_log_path is None:
            out_dir = str(getattr(self.args, "logging_dir", "./outputs") or "./outputs")
            base = os.path.join(out_dir, "per_example_train")
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                return None
            self._pe_log_dir = base
            self._pe_log_path = os.path.join(base, "stream.jsonl")
        return self._pe_log_path

    @torch.no_grad()
    def _collect_per_example_from_batch(
        self,
        inputs: Dict[str, Any],
        shift_logits: torch.Tensor,  # (B, S-1, V)
        shift_labels: torch.Tensor,  # (B, S-1)
        fork_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        # Per-token loss
        vocab = shift_logits.size(-1)
        per_token = F.cross_entropy(
            shift_logits.float().view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=self._ignore_index,
            reduction="none",
        ).view_as(shift_labels).float()

        # Fork weighting (same as in loss), default alpha on non-fork tokens
        if fork_mask is None:
            shift_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        else:
            shift_mask = fork_mask[..., 1:].contiguous().bool()
        token_w = torch.where(shift_mask, torch.ones_like(per_token), torch.full_like(per_token, self._fork_alpha))

        valid = (shift_labels != self._ignore_index)
        token_w = token_w * valid

        per_ex_sum = (per_token * token_w).sum(dim=1, dtype=torch.float32)
        per_ex_cnt = valid.sum(dim=1, dtype=torch.float32).clamp_min(1)
        per_ex_loss = (per_ex_sum / per_ex_cnt).detach().cpu()

        # Try to find example identifiers carried in the batch
        id_keys = ("id", "ids", "example_id", "example_ids", "idx", "index")
        ex_ids: Optional[List[Any]] = None
        for k in id_keys:
            if k in inputs:
                try:
                    v = inputs[k]
                    if isinstance(v, (list, tuple)):
                        ex_ids = list(v)
                    elif torch.is_tensor(v):
                        ex_ids = v.detach().cpu().tolist()
                    else:
                        # don't know; skip
                        pass
                except Exception:
                    pass
                if ex_ids is not None:
                    break

        # Optional decoded snippets to help locate offending cases
        snippets: Optional[List[str]] = None
        if self._pe_log_text and "input_ids" in inputs:
            try:
                input_ids = inputs["input_ids"]
                if torch.is_tensor(input_ids):
                    # Extract only completion span based on labels mask
                    mask = (shift_labels != self._ignore_index)
                    # Align to input_ids without the last token (because of shift)
                    ii = input_ids[:, :-1]
                    seqs = [ii[b][mask[b]].tolist() if mask[b].any() else ii[b].tolist() for b in range(ii.size(0))]
                    tok = getattr(self, "processing_class", None)
                    if tok is not None and hasattr(tok, "decode"):
                        snippets = [tok.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)[:256] for s in seqs]
                    else:
                        snippets = None
                else:
                    snippets = None
            except Exception:
                snippets = None

        return {
            "loss": per_ex_loss.tolist(),
            "count": per_ex_cnt.detach().cpu().tolist(),
            "ids": ex_ids,
            "snippets": snippets,
        }

    def compute_loss(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        fork_mask = inputs.pop(self._fork_mask_key, None)

        labels = inputs.get("labels", None)
        outputs = model(**inputs)

        if labels is None:
            raise ValueError("ForkWeighedLossTrainer requires 'labels' in inputs to compute loss.")

        logit_scale = float(model.config.logits_scaling) if hasattr(model.config, "logits_scaling") else 1.0 # type: ignore[arg-type]

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = self._fork_weighted_shifted_loss(logits, labels, fork_mask, logit_divider=logit_scale)

        # Per-example logging (opt-in)
        if self._pe_log_topk > 0:
            try:
                if logit_scale != 0:
                    logits_for_log = logits / logit_scale
                else:
                    logits_for_log = logits
                shift_logits = logits_for_log[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                details = self._collect_per_example_from_batch(inputs, shift_logits, shift_labels, fork_mask)

                # Rank by loss and take top-k exceeding threshold
                losses = details.get("loss", [])
                if isinstance(losses, list) and losses:
                    try:
                        import numpy as _np  # optional
                        arr = _np.asarray(losses, dtype=float)
                        k = max(1, min(self._pe_log_topk, arr.size))
                        idxs = _np.argpartition(-arr, k - 1)[:k]
                        worst = sorted([(int(i), float(arr[i])) for i in idxs if float(arr[i]) >= self._pe_log_min_loss], key=lambda x: -x[1])
                    except Exception:
                        # Fallback: pure Python partial sort
                        pairs = [(i, float(l)) for i, l in enumerate(losses) if float(l) >= self._pe_log_min_loss]
                        pairs.sort(key=lambda x: -x[1])
                        worst = pairs[: max(1, min(self._pe_log_topk, len(pairs)))]
                    if worst:
                        path = self._ensure_pe_log_path()
                        payloads = []
                        step = int(getattr(self.state, "global_step", 0)) if hasattr(self, "state") else None
                        for i, l in worst:
                            item = {
                                "step": step,
                                "rank": int(i),
                                "loss": float(l),
                                "tok_count": (details.get("count") or [None])[i] if i < len(details.get("count") or []) else None,
                                "id": (details.get("ids") or [None])[i] if details.get("ids") is not None and i < len(details.get("ids") or []) else None,
                            }
                            if self._pe_log_text:
                                snips = details.get("snippets") or []
                                item["snippet"] = (snips[i] if i < len(snips) else None)
                            payloads.append(item)

                        # also stash for callbacks / interactive inspection
                        self._last_batch_per_example = {
                            "step": step,
                            "topk": payloads,
                        }

                        if path is not None:
                            try:
                                with open(path, "a", encoding="utf-8") as f:
                                    for p in payloads:
                                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                            except Exception:
                                pass
                        else:
                            # Fallback: print succinctly
                            try:
                                print("[per-ex] step=", step, "worst=", payloads)
                            except Exception:
                                pass
            except Exception:
                # Never break training on logging errors
                self._last_batch_per_example = None
        return (loss, outputs) if return_outputs else loss

    def _fork_weighted_shifted_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            fork_mask: Optional[torch.Tensor],
            logit_divider: float = 1.0
    ) -> torch.Tensor:
        if logit_divider != 0:
            logits = logits / logit_divider

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab = shift_logits.size(-1)

        per_token = F.cross_entropy(
            shift_logits.float().view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=self._ignore_index,
            reduction="none",
        ).view_as(shift_labels).float()

        # Robust mask handling: if no fork_mask provided, treat as all False (apply alpha everywhere)
        if fork_mask is None:
            shift_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        else:
            shift_mask = fork_mask[..., 1:].contiguous().bool()
        token_w = torch.where(shift_mask, torch.ones_like(per_token), torch.full_like(per_token, self._fork_alpha))

        valid = (shift_labels != self._ignore_index)
        token_w = token_w * valid
        num = (per_token * token_w).sum(dtype=torch.float32)
        den = valid.sum(dtype=torch.float32).clamp_min(1)
        loss = num / den

        return loss