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

from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, cast
from dataclasses import dataclass

import torch
from trl import SFTTrainer

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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_lr = float(base_lr)
        self._specs: List[GroupSpec] = list(group_specs) if group_specs is not None else []

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


__all__ = [
    "GroupSpec",
    "build_param_groups_from_specs",
    "LayerwiseLRTrainer",
    # helpers
    "build_layered_suffixes",
    "build_target_modules",
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
