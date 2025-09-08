import argparse
import json
import gc
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig

# Local imports (mirror existing scripts)
from data_collators import DataCollatorForLastCompletionOnlyLM
from callbacks import MetricCalculator, PerExampleEvalLogger, GradTensorBoardLogger, LogSamplerOrder
from trainer import (
    GroupSpec,
    LayerwiseLRTrainer,
    ForkWeighedLossTrainer,
    build_target_modules,
    apply_weight_rescale,
)


# ------------------------------
# Config loading with inheritance
# ------------------------------

def _deep_merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Default behavior: lists are replaced. Special case: if merging a list of
    group dicts under key 'groups' (e.g., layerwise_lr.groups), merge by
    matching 'name' field so overrides can tweak a single group without
    discarding others.
    """

    def _merge_named_list(base_list: List[Any], override_list: List[Any], key: str = "name") -> List[Any]:
        # Build index from base by key when elements are dicts with the key
        result: List[Any] = [deepcopy(x) for x in base_list]
        index: Dict[Any, int] = {}
        for i, el in enumerate(result):
            if isinstance(el, dict) and key in el:
                index[el[key]] = i
        for el in override_list:
            if isinstance(el, dict) and key in el and el[key] in index:
                # deep-merge into existing
                i = index[el[key]]
                result[i] = _deep_merge(result[i], el)  # type: ignore[arg-type]
            else:
                # append new or non-dict
                result.append(deepcopy(el))
        return result

    out = dict(parent)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        elif isinstance(v, list) and isinstance(out.get(k), list) and k == "groups":
            out[k] = _merge_named_list(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    cfg = _load_yaml(path)
    extends = cfg.get("extends")
    if extends:
        parent_path = extends
        if not os.path.isabs(parent_path):
            parent_path = os.path.normpath(os.path.join(os.path.dirname(path), parent_path))
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, {k: v for k, v in cfg.items() if k != "extends"})
    # After merge, apply environment variable interpolation ${VAR}
    # Build a mapping for dotted aliases (stage.name -> STAGE_NAME, etc.)
    alias_map = {
        "stage.name": os.environ.get("STAGE_NAME"),
        "stage.index": os.environ.get("STAGE_INDEX"),
        "stage.dir": os.environ.get("STAGE_DIR"),
        "run.index": os.environ.get("RUN_INDEX"),
        "run.dir": os.environ.get("RUN_DIR"),
        "repo.root": os.environ.get("REPO_ROOT"),
        "pipeline.dir": os.environ.get("PIPELINE_DIR"),
    }

    def _interp(obj: Any) -> Any:
        if isinstance(obj, str):
            out = obj
            for _ in range(3):
                # replace ${VAR} tokens with environment values if present
                import re
                def repl(m: re.Match[str]) -> str:
                    key = m.group(1)
                    # Try dotted alias first
                    if key in alias_map and alias_map[key] is not None:
                        return str(alias_map[key])
                    return os.environ.get(key, m.group(0))
                new_out = re.sub(r"\$\{([^}]+)\}", repl, out)
                if new_out == out:
                    break
                out = new_out
            return out
        if isinstance(obj, list):
            return [_interp(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_interp(list(obj)))
        if isinstance(obj, dict):
            return {k: _interp(v) for k, v in obj.items()}
        return obj
    return _interp(cfg)


# ------------------------------
# Helpers
# ------------------------------

def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _expand_layers(layer_spec: Union[Dict[str, int], Sequence[int], range, None]) -> Optional[List[int]]:
    if layer_spec is None:
        return None
    if isinstance(layer_spec, dict) and "start" in layer_spec and "end" in layer_spec:
        start = int(layer_spec["start"]) 
        end = int(layer_spec["end"])
        if end < start:
            raise ValueError(f"Invalid layer range: start={start}, end={end}")
        return list(range(start, end + 1))
    if isinstance(layer_spec, range):
        return list(layer_spec)
    # assume list-like of ints
    return [int(x) for x in layer_spec]  # type: ignore[arg-type]


def _normpath_join(base: Optional[str], rel: str) -> str:
    if base:
        return os.path.normpath(os.path.join(base, rel))
    return os.path.normpath(rel)


def _get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


# ------------------------------
# Token additions + grad boost
# ------------------------------

def add_and_initialize_tokens(
    tokenizer,
    model: torch.nn.Module,
    token_inits: List[Tuple[str, str]],
) -> Tuple[List[int], List[int]]:
    """Add tokens (if missing) and initialize embeddings from init strings.

    Returns (new_token_ids, all_token_ids_for_list).
    Only new tokens are initialized; existing tokens remain unchanged.
    """
    # Add tokens that don't exist yet (HF tokenizer ignores duplicates)
    tokens_to_add = [t for t, _ in token_inits]
    pre_vocab_size = len(tokenizer)
    added_count = tokenizer.add_tokens(tokens_to_add)
    if added_count:
        try:
            resize_fn = getattr(model, "resize_token_embeddings", None)
            if callable(resize_fn):
                resize_fn(len(tokenizer), mean_resizing=False)
        except Exception:
            pass

    # Determine which tokens are newly added
    new_token_ids: List[int] = []
    for tok, _ in token_inits:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id >= pre_vocab_size:
            new_token_ids.append(tok_id)

    # Initialize newly added tokens from the mean of init string token embeddings
    emb_weight = None
    try:
        emb_getter = getattr(model, "get_input_embeddings", None)
        emb_mod = emb_getter() if callable(emb_getter) else None
        emb_weight = getattr(emb_mod, "weight", None)
    except Exception:
        emb_weight = None
    if isinstance(emb_weight, torch.Tensor):
        with torch.no_grad():
            fallback_vec = emb_weight.mean(dim=0)
            for tok, init_str in token_inits:
                tok_id = tokenizer.convert_tokens_to_ids(tok)
                if tok_id in new_token_ids:
                    seed_ids: List[int] = tokenizer.encode(init_str, add_special_tokens=False)
                    if len(seed_ids) == 0:
                        vec = fallback_vec
                    else:
                        vec = emb_weight[seed_ids].mean(dim=0)
                    emb_weight[tok_id].copy_(vec)

    # Build list of all tokens referenced (existing or new), deduped
    all_token_ids: List[int] = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok, _ in token_inits
    ]
    all_token_ids = [tid for i, tid in enumerate(all_token_ids) if tid != -1 and tid not in all_token_ids[:i]]

    return new_token_ids, all_token_ids


def register_token_grad_boost(
    model: torch.nn.Module,
    token_ids: List[int],
    factor: float = 10.0,
    apply_to_input: bool = True,
    apply_to_output: bool = True,
) -> None:
    """Scale gradients for specified token rows in input embeddings and/or lm_head."""
    if not token_ids:
        return

    # Input embeddings
    if apply_to_input:
        try:
            emb_getter = getattr(model, "get_input_embeddings", None)
            emb_mod = emb_getter() if callable(emb_getter) else None
            emb_weight = emb_mod.weight if emb_mod is not None and hasattr(emb_mod, "weight") else None  # type: ignore[assignment]
            if emb_weight is None:
                raise RuntimeError("No input embedding weight found")
            def scale_rows(grad: torch.Tensor):
                if grad is None or grad.numel() == 0:
                    return grad
                ids_tensor = torch.tensor(token_ids, device=grad.device, dtype=torch.long)
                scaled = grad.index_select(0, ids_tensor) * factor
                grad.index_copy_(0, ids_tensor, scaled)
                return grad

            emb_weight.register_hook(scale_rows)
        except Exception:
            pass

    # Output embeddings (lm_head) if not tied
    if apply_to_output:
        lm_head = None
        get_out = getattr(model, "get_output_embeddings", None)
        if callable(get_out):
            try:
                lm_head = get_out()
            except Exception:
                lm_head = None
        if lm_head is None:
            lm_head = getattr(model, "lm_head", None)
        try:
            in_emb_get = getattr(model, "get_input_embeddings", None)
            in_emb_mod = in_emb_get() if callable(in_emb_get) else None
            in_emb_weight = getattr(in_emb_mod, "weight", None)
            lm_w = getattr(lm_head, "weight", None)
            if isinstance(lm_w, torch.Tensor):
                # Only apply if not weight-tied
                if not (isinstance(in_emb_weight, torch.Tensor) and lm_w.data_ptr() == in_emb_weight.data_ptr()):
                    def scale_rows_lm(grad: torch.Tensor):
                        if grad is None or grad.numel() == 0:
                            return grad
                        ids_tensor = torch.tensor(token_ids, device=grad.device, dtype=torch.long)
                        scaled = grad.index_select(0, ids_tensor) * factor
                        grad.index_copy_(0, ids_tensor, scaled)
                        return grad

                    lm_w.register_hook(scale_rows_lm)
        except Exception:
            pass


# ------------------------------
# PEFT target resolution
# ------------------------------

def resolve_target_modules(cfg: Dict[str, Any]) -> List[str]:
    peft = cfg.get("peft", {})
    lwise = cfg.get("layerwise_lr", {})

    include_modules: List[str] = []
    layered_specs: List[Tuple[Iterable[int], str]] = []

    # 1) Fully-enabled modules
    for mod in peft.get("modules", []) or []:
        include_modules.append(str(mod))

    # Helper: map module -> list of layer lists from groups
    group_layers_by_suffix: Dict[str, List[List[int]]] = {}
    for grp in lwise.get("groups", []) or []:
        layers = _expand_layers(grp.get("layers"))
        suffixes = grp.get("suffixes", []) or []
        if layers is None:
            continue
        for sfx in suffixes:
            group_layers_by_suffix.setdefault(sfx, []).append(layers)

    # 2) layered_modules explicit / auto
    layered_modules = peft.get("layered_modules", {}) or {}
    if isinstance(layered_modules, list):
        # Backward compat: list-of-singleton-maps
        lm_dict: Dict[str, Any] = {}
        for item in layered_modules:
            if isinstance(item, dict):
                lm_dict.update(item)
        layered_modules = lm_dict

    for mod, spec in layered_modules.items():
        layers_spec = spec.get("layers") if isinstance(spec, dict) else None
        if layers_spec == "auto":
            # Use all group ranges matching this module
            for layers in group_layers_by_suffix.get(mod, []):
                layered_specs.append((layers, mod))
        else:
            layers = _expand_layers(layers_spec)
            if layers:
                layered_specs.append((layers, mod))

    # 3) force_enable from groups
    for grp in lwise.get("groups", []) or []:
        if not grp.get("force_enable"):
            continue
        layers = _expand_layers(grp.get("layers"))
        suffixes = grp.get("suffixes", []) or []
        if layers is None:
            continue
        for sfx in suffixes:
            layered_specs.append((layers, sfx))

    # Build final list via helper (dedupe)
    tm = build_target_modules(
        include_attention=False,
        include_mlp=False,
        include_modules=include_modules,
        layered=layered_specs,
    )
    # Dedupe
    seen = set()
    deduped: List[str] = []
    for x in tm:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


# ------------------------------
# Batch size computation
# ------------------------------

def compute_per_device_batch(cfg: Dict[str, Any]) -> int:
    train = cfg.get("training", {})
    eff = int(train.get("effective_batch_size", 1))
    ga = int(train.get("gradient_accumulation_steps", 1))
    bc = train.get("batch_compute", {}) or {}
    min_per = int(bc.get("minimum_per_device", 1))

    # Detect devices
    device_count = torch.cuda.device_count() if bc.get("device_count", "auto") == "auto" else int(bc["device_count"])  # type: ignore[index]
    denom = max(1, ga * max(1, device_count))
    if eff % denom != 0:
        print(f"[Batch] Warning: effective_batch_size={eff} not divisible by GA*devices={denom}. Flooring.")
    per_device = max(min_per, eff // denom)
    if per_device < 1:
        print(f"[Batch] Warning: computed per-device batch < 1; forcing to {min_per}. This changes effective total batch.")
    return per_device


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generic trainer with YAML config")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Model load params
    model_cfg = cfg.get("model", {})
    dtype = _dtype_from_str(str(model_cfg.get("dtype", "bfloat16")))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_cfg.get("model_name")),
        max_seq_length=int(model_cfg.get("max_seq_length", 2048)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
        dtype=dtype,
        attn_implementation=str(model_cfg.get("attn_implementation", "eager")),
    )

    # Dataset build
    ds_cfg = cfg.get("dataset", {})
    text_field = str(ds_cfg.get("text_field", "prompt"))

    if ds_cfg.get("train_files") and ds_cfg.get("eval_files"):
        train_files = [str(p) for p in ds_cfg["train_files"]]
        eval_files = [str(p) for p in ds_cfg["eval_files"]]
    else:
        base = ds_cfg.get("base_path")
        stats = ds_cfg.get("stats", []) or []
        train_tpl = str(ds_cfg.get("train_template", "{stat}.train.jsonl"))
        eval_tpl = str(ds_cfg.get("eval_template", "{stat}.eval.jsonl"))
        train_files = [_normpath_join(base, train_tpl.format(stat=s)) for s in stats]
        eval_files = [_normpath_join(base, eval_tpl.format(stat=s)) for s in stats]

    dataset = load_dataset("json", data_files=train_files, split="train")
    eval_dataset = load_dataset("json", data_files=eval_files, split="train")

    # # Attach a stable example_id early so it can be threaded into batches
    # try:
    #     def _add_id(ex, idx):
    #         val = None
    #         try:
    #             # Prefer explicit id if present
    #             if isinstance(ex.data, dict) and "id" in ex.data and ex.data["id"] is not None:
    #                 val = ex.data["id"]
    #             elif isinstance(ex.data, dict) and "example_id" in ex.data and ex.data["example_id"] is not None:
    #                 val = ex.data["example_id"]
    #         except Exception:
    #             val = None
    #         if val is None:
    #             val = idx
    #         return {"example_id": val}

    #     dataset = dataset.map(_add_id, with_indices=True)
    #     eval_dataset = eval_dataset.map(_add_id, with_indices=True)
    # except Exception:
    #     pass

    # Collator
    coll_cfg = cfg.get("collator", {})
    completion_marker = str(coll_cfg.get("completion_start_marker", '="'))
    loss_cfg_for_coll = cfg.get("loss", {}) or {}
    collator = DataCollatorForLastCompletionOnlyLM(
        completion_marker,
        tokenizer=tokenizer,
        fork_mask_key=str(loss_cfg_for_coll.get("fork_mask_key", "fork_mask")) if bool(loss_cfg_for_coll.get("use_fork_weighed", loss_cfg_for_coll.get("fork_weighed_enabled", False))) else None,
        auto_full_mask=bool(loss_cfg_for_coll.get("force_full_mask", False)),
        mask_generator=loss_cfg_for_coll.get("mask_generator", None),
    )

    # Optional: token additions and gradient boosts
    tokens_cfg = cfg.get("tokens", {}) or {}
    if tokens_cfg.get("pad_to_eos"):
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    token_inits_cfg = tokens_cfg.get("add_and_init") or tokens_cfg.get("additions") or []
    token_inits: List[Tuple[str, str]] = []
    for item in token_inits_cfg:
        if isinstance(item, dict) and "token" in item and "init" in item:
            token_inits.append((str(item["token"]), str(item["init"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            token_inits.append((str(item[0]), str(item[1])))

    new_token_ids: List[int] = []
    all_token_ids: List[int] = []
    if token_inits:
        new_token_ids, all_token_ids = add_and_initialize_tokens(tokenizer, model, token_inits)

    # Stash grad-boost config to apply AFTER PEFT wrapping
    gb = tokens_cfg.get("grad_boost") or {}

    # Metrics
    met_cfg = cfg.get("metrics", {})
    exp_field = met_cfg.get("expected_value_field")
    target_attr_path = met_cfg.get("target_attr_path")
    expected_values: Optional[List[str]] = None
    target_attrs: Optional[List[str]] = None
    eval_ids: Optional[List[Any]] = None
    try:
        if exp_field:
            expected_values = [str(ex.get(exp_field, "")) if isinstance(ex, dict) else "" for ex in eval_dataset]
        if target_attr_path:
            tmp: List[str] = []
            for ex in eval_dataset:
                if isinstance(ex, dict):
                    val = _get_nested(ex, str(target_attr_path))
                    tmp.append(str(val) if val is not None else "")
                else:
                    tmp.append("")
            target_attrs = tmp
        # capture example ids for per-datapoint logging
        try:
            eval_ids = [
                (ex.get("example_id") if isinstance(ex, dict) and ex.get("example_id") is not None else ex.get("id", None))
                if isinstance(ex, dict) else None
                for ex in eval_dataset
            ]
        except Exception:
            eval_ids = None
    except Exception:
        # Fallback to None if unexpected structure
        expected_values = None
        target_attrs = None
        eval_ids = None

    metric_calculator = MetricCalculator(tokenizer, expected_values=expected_values, target_attrs=target_attrs, example_ids=eval_ids)

    # PEFT / LoRA
    peft_cfg = cfg.get("peft", {})
    target_modules = resolve_target_modules(cfg)

    rank = int(peft_cfg.get("r", 8))
    alpha = peft_cfg.get("lora_alpha", 8)
    if (alpha == "rank"):
        alpha = rank
    else:
        alpha = int(alpha)

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),  # type: ignore[arg-type]
        use_gradient_checkpointing=peft_cfg.get("use_gradient_checkpointing", "unsloth"),  # type: ignore[arg-type]
        random_state=int(peft_cfg.get("random_state", 3407)),
        modules_to_save=[str(m) for m in peft_cfg.get("modules_to_save", [])],
    )

    if peft_cfg.get("force_requires_grad_on_embeddings", True):
        try:
            model.get_input_embeddings().weight.requires_grad_(True)
        except Exception as e:
            print(f"[Warning] Could not force requires_grad on embeddings: {e}")

    # Apply gradient boost hooks now that PEFT has wrapped/replaced modules
    if gb and (new_token_ids or all_token_ids):
        factor = float(gb.get("factor", 10.0))
        target = str(gb.get("target", "all")).lower()  # 'new' or 'all'
        apply_to = gb.get("apply_to", ["input", "output"]) or []
        apply_to_input = "input" in apply_to
        apply_to_output = "output" in apply_to
        ids = new_token_ids if target == "new" else all_token_ids
        if ids:
            register_token_grad_boost(
                model,
                ids,
                factor=factor,
                apply_to_input=apply_to_input,
                apply_to_output=apply_to_output,
            )
            print(f"[Tokens] Registered grad boost on {len(ids)} token(s); factor={factor}; input={apply_to_input}, output={apply_to_output}")

    # Layerwise groups for optimizer
    lwise = cfg.get("layerwise_lr", {})
    base_lr = float(lwise.get("base_lr", 3e-5))
    group_specs: List[GroupSpec] = []
    for grp in lwise.get("groups", []) or []:
        layers = _expand_layers(grp.get("layers"))
        gs = GroupSpec(
            name=str(grp.get("name", "group")),
            suffixes=[str(s) for s in (grp.get("suffixes") or [])],
            layers=layers,
            lr=float(grp["lr"]) if "lr" in grp else None,
            lr_scale=float(grp["lr_scale"]) if "lr_scale" in grp else None,
            weight_decay=float(grp["weight_decay"]) if "weight_decay" in grp else None,
            weight_rescale=float(grp["weight_rescale"]) if "weight_rescale" in grp else None,
        )
        group_specs.append(gs)

    # Apply one-time weight rescale before PEFT wrapping so it affects base weights
    try:
        apply_weight_rescale(model, group_specs)
    except Exception:
        pass

    # Force require_grad on groups without PEFT
    force_grad_groups = [grp for grp in lwise.get("groups", []) if grp.get("force_require_grad", False)]
    if force_grad_groups:
        for grp in force_grad_groups:
            layers = _expand_layers(grp.get("layers"))
            suffixes = grp.get("suffixes", []) or []
            if layers is None:
                # For non-layered, like embed
                for suffix in suffixes:
                    for name, param in model.named_parameters():
                        if suffix in name:
                            param.requires_grad = True
            else:
                for layer in layers:
                    for suffix in suffixes:
                        pattern = f"layers.{layer}.{suffix}"
                        for name, param in model.named_parameters():
                            if pattern in name:
                                param.requires_grad = True

    # Batch sizes
    per_device_train_bs = compute_per_device_batch(cfg)

    # SFT args
    train_cfg = cfg.get("training", {})
    dataset_seed = train_cfg.get("dataset_seed", None)

    # Infer precision from model dtype (avoid mismatch)
    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16

    # Ensure TensorBoard uses run_name: set a stable logging_dir
    from datetime import datetime

    # Optional swipe name to differentiate runs/paths
    swipe_name = str(cfg.get("swipe_name", "")).strip()

    # Output directory (optionally nested by swipe)
    output_dir = str(train_cfg.get("output_dir", "./outputs"))
    base_run_name = str(train_cfg.get("run_name", "run"))
    if swipe_name:
        output_dir = os.path.join(output_dir, swipe_name)
    else:
        output_dir = os.path.join(output_dir, base_run_name)

    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    run_name = f"{timestamp}"

    # Logging dir defaults under the (possibly swipe-namespaced) output_dir
    logging_dir = str(train_cfg.get("logging_dir", os.path.join(output_dir, "runs")))
    logging_dir = os.path.join(logging_dir, swipe_name, run_name)

    sft_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field=text_field,
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        dataset_num_proc=int(train_cfg.get("dataset_num_proc", 1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        lr_scheduler_kwargs=train_cfg.get("lr_scheduler_kwargs", {}),
        learning_rate=float(train_cfg.get("learning_rate", base_lr)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        optim=str(train_cfg.get("optim", "adamw_torch")),
        adam_beta1=float(train_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(train_cfg.get("adam_beta2", 0.999)),
        max_steps=int(train_cfg.get("max_steps", 0)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=str(train_cfg.get("report_to", "none")),
        run_name=run_name,
        logging_dir=logging_dir,
        save_strategy=str(train_cfg.get("save_strategy", "no")),
        save_steps=int(train_cfg.get("save_steps", 0)),
        save_only_model=bool(train_cfg.get("save_only_model", True)),
        eval_strategy=str(train_cfg.get("eval_strategy", "no")),
        eval_steps=int(train_cfg.get("eval_steps", 0)),
        eval_on_start=bool(train_cfg.get("eval_on_start", False)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", per_device_train_bs)),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(peft_cfg.get("use_gradient_checkpointing", "unsloth") is not False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_empty_cache_steps=train_cfg.get("torch_empty_cache_steps", None),
    )

    # Persist hyperparameters (+ LoRA) to JSON in the logging directory
    hparams: Dict[str, Any] = {}

    try:
        os.makedirs(logging_dir, exist_ok=True)

        def _json_safe(v: Any) -> Any:
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    return v
                # Enums (e.g., lr scheduler type) -> string value
                if hasattr(v, "value"):
                    return _json_safe(getattr(v, "value"))
                # Tensors -> summary string
                if isinstance(v, torch.Tensor):
                    return {"tensor": list(v.size())}
                # Paths and other objects -> string
                if isinstance(v, (list, tuple)):
                    return [_json_safe(x) for x in v]
                if isinstance(v, dict):
                    return {str(k): _json_safe(val) for k, val in v.items()}
                return str(v)
            except Exception:
                return str(v)

        # Trainer/TrainingArguments (SFTConfig)
        try:
            trainer_args_dict = sft_args.to_dict()  # type: ignore[attr-defined]
        except Exception:
            try:
                # transformers >=4.44 has to_sanitized_dict
                trainer_args_dict = sft_args.to_sanitized_dict()  # type: ignore[attr-defined]
            except Exception:
                trainer_args_dict = {k: getattr(sft_args, k) for k in dir(sft_args) if not k.startswith("_") and not callable(getattr(sft_args, k, None))}
        trainer_args_dict = {str(k): _json_safe(v) for k, v in dict(trainer_args_dict).items()}

        # LoRA/PEFT details
        lora_info: Dict[str, Any] = {
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": float(peft_cfg.get("lora_dropout", 0.0)),
            "use_gradient_checkpointing": _json_safe(peft_cfg.get("use_gradient_checkpointing", "unsloth")),
            "random_state": int(peft_cfg.get("random_state", 3407)),
            "modules_to_save": [str(m) for m in peft_cfg.get("modules_to_save", [])],
            "target_modules": list(target_modules),
            "force_requires_grad_on_embeddings": bool(peft_cfg.get("force_requires_grad_on_embeddings", True)),
            "dtype": str(model_cfg.get("dtype", "bfloat16")),
        }

        # Layerwise LR details
        layerwise_dump = {
            "base_lr": base_lr,
            "groups": [
                {
                    "name": gs.name,
                    "suffixes": list(gs.suffixes),
                    "layers": list(gs.layers) if gs.layers is not None else None,
                    "lr": gs.lr,
                    "lr_scale": gs.lr_scale,
                    "weight_decay": gs.weight_decay,
                }
                for gs in group_specs
            ],
        }

        # Loss/trainer selection (optional)
        loss_cfg = cfg.get("loss", {}) or {}
        use_fork_loss = bool(loss_cfg.get("use_fork_weighed", loss_cfg.get("fork_weighed_enabled", False)))
        fork_alpha = float(loss_cfg.get("fork_alpha", 1.0))
        fork_mask_key = str(loss_cfg.get("fork_mask_key", "fork_mask"))
        fork_ignore_index = -100 #int(loss_cfg.get("ignore_index", -100))

        hparams = {
            "run": {
                "swipe_name": swipe_name,
                "run_name": run_name,
                "output_dir": output_dir,
                "logging_dir": logging_dir,
            },
            "trainer": trainer_args_dict,
            "peft": lora_info,
            "layerwise_lr": layerwise_dump,
            "loss": {
                "trainer_class": ("ForkWeighedLossTrainer" if use_fork_loss else "LayerwiseLRTrainer"),
                "use_fork_weighed": use_fork_loss,
                "fork_alpha": fork_alpha,
                "fork_mask_key": fork_mask_key,
                "ignore_index": fork_ignore_index,
                "fork_alpha_schedule": (loss_cfg.get("fork_alpha_schedule", None) or None),
            },
        }

        with open(os.path.join(logging_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(hparams, f, ensure_ascii=False, indent=2)
        print(f"[HParams] Wrote hyperparameters to {os.path.join(logging_dir, 'hparams.json')}")
    except Exception as e:
        print(f"[HParams] Failed to write hparams: {e}")

    # Select trainer class based on YAML loss.toggle
    loss_cfg = cfg.get("loss", {}) or {}
    use_fork_loss = bool(loss_cfg.get("use_fork_weighed", loss_cfg.get("fork_weighed_enabled", False)))
    trainer_cls = ForkWeighedLossTrainer if use_fork_loss else LayerwiseLRTrainer

    common_kwargs: Dict[str, Any] = dict(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        base_lr=base_lr,
        group_specs=group_specs,
        compute_metrics=metric_calculator.compute_metrics,
        preprocess_logits_for_metrics=metric_calculator.keep_argmax,
        args=sft_args,
        dataset_seed=(int(dataset_seed) if dataset_seed is not None else None),
    )

    if trainer_cls is ForkWeighedLossTrainer:
        common_kwargs.update({
            "fork_alpha": float(loss_cfg.get("fork_alpha", 1.0)),
            "fork_mask_key": str(loss_cfg.get("fork_mask_key", "fork_mask")),
            "ignore_index": int(loss_cfg.get("ignore_index", -100)),
        })

    trainer = trainer_cls(**common_kwargs)

    # Add per-example eval logger (writes JSONL files, not TensorBoard)
    try:
        per_ex_logger = PerExampleEvalLogger(metric_calculator, output_dir=logging_dir)
        trainer.add_callback(per_ex_logger)
    except Exception:
        pass

    # Optional: gradient TensorBoard logging per layer/kind
    try:
        grad_log_cfg = cfg.get("gradient_logging", {}) or {}
        if bool(grad_log_cfg.get("enabled", False)):
            glog_dir = str(grad_log_cfg.get("logging_dir", logging_dir))
            kinds = grad_log_cfg.get("include_kinds", ["self_attn", "mlp"]) or ["self_attn", "mlp"]
            tag_prefix = str(grad_log_cfg.get("tag_prefix", "grads"))
            grad_cb = GradTensorBoardLogger(logging_dir=glog_dir, include_kinds=kinds, tag_prefix=tag_prefix)
            trainer.add_callback(grad_cb)
    except Exception:
        pass

    # Optional: fork_alpha scheduler + TB logging
    try:
        loss_cfg = cfg.get("loss", {}) or {}
        sched_cfg = loss_cfg.get("fork_alpha_schedule", None)
        if trainer_cls is ForkWeighedLossTrainer and sched_cfg:
            from callbacks import ForkAlphaScheduler
            sched_type = str(sched_cfg.get("type", "constant").split("|")[0]).strip() if isinstance(sched_cfg.get("type"), str) else str(sched_cfg.get("type", "constant"))
            # If user writes "linear | cosine | constant", pick from YAML elsewhere; here assume explicit like "linear"
            # We'll accept exact string and fallback to constant.
            if sched_type in ("linear", "cosine", "constant"):
                base_alpha = float(loss_cfg.get("fork_alpha", 1.0))
                def _set_alpha(a: float) -> None:
                    try:
                        setattr(trainer, "_fork_alpha", float(a))
                    except Exception:
                        pass
                scheduler_cb = ForkAlphaScheduler(
                    base_alpha=base_alpha,
                    sched_type=sched_type,
                    args=(sched_cfg.get("args", None) or {}),
                    on_update=_set_alpha,
                    logging_dir=logging_dir,
                    use_trainer_logging_dir=True,
                    filename_suffix=".fork_alpha",
                )
                try:
                    trainer.add_callback(scheduler_cb)
                except Exception:
                    pass
    except Exception:
        pass

    # Optional: max_grad_norm scheduler + TB logging
    try:
        train_cfg = cfg.get("training", {}) or {}
        mg_sched = train_cfg.get("max_grad_norm_schedule", None)
        if mg_sched:
            from callbacks import MaxGradNormScheduler
            mg_type = str(mg_sched.get("type", "constant").split("|")[0]).strip() if isinstance(mg_sched.get("type"), str) else str(mg_sched.get("type", "constant"))
            if mg_type in ("linear", "cosine", "constant"):
                base_mg = float(train_cfg.get("max_grad_norm", 1.0))
                def _set_mg(v: float) -> None:
                    try:
                        setattr(trainer.args, "max_grad_norm", float(v))
                    except Exception:
                        pass
                mg_cb = MaxGradNormScheduler(
                    base_value=base_mg,
                    sched_type=mg_type,
                    args=(mg_sched.get("args", None) or {}),
                    on_update=_set_mg,
                    logging_dir=logging_dir,
                    use_trainer_logging_dir=True,
                    filename_suffix=".grad_clip",
                )
                try:
                    trainer.add_callback(mg_cb)
                except Exception:
                    pass
    except Exception:
        pass

    #trainer.add_callback(LogSamplerOrder(trainer))

    trainer.train(resume_from_checkpoint=False)

    # Capture the last available eval metrics from log history
    last_eval_metrics: Optional[Dict[str, Any]] = None
    try:
        def _coerce_jsonable(v: Any) -> Any:
            # Best-effort conversion to JSON-friendly values
            try:
                if isinstance(v, (int, float, str)) or v is None:
                    return v
                # torch / numpy scalars
                return float(v)
            except Exception:
                try:
                    return str(v)
                except Exception:
                    return None

        log_hist = getattr(trainer.state, "log_history", [])
        # Find the last dict containing any eval_* keys
        last_eval_entry: Optional[Dict[str, Any]] = None
        for entry in reversed(log_hist):
            if isinstance(entry, dict) and any(k.startswith("eval_") for k in entry.keys()):
                last_eval_entry = entry
                break
        if last_eval_entry:
            # Keep eval_* keys and a couple of useful metadata if present
            filtered = {k: _coerce_jsonable(v) for k, v in last_eval_entry.items() if k.startswith("eval_") or k in ("step", "epoch")}
            if filtered:
                last_eval_metrics = filtered
    except Exception:
        last_eval_metrics = None

    # Ensure an evaluation at the end if it didn't occur naturally at the final step
    try:
        log_hist = getattr(trainer.state, "log_history", [])
        def _is_eval_entry(x: Any) -> bool:
            return isinstance(x, dict) and any(k.startswith("eval_") for k in x.keys())

        eval_steps = [int(x.get("step", -1)) for x in log_hist if _is_eval_entry(x)]
        last_eval_step = max(eval_steps) if eval_steps else -1
        final_step = int(getattr(trainer.state, "global_step", 0))
        if last_eval_step < final_step and eval_dataset is not None:
            print("[Eval] No eval at final step; running a final evaluation now...")
            final_metrics = trainer.evaluate()
            # Short summary print
            try:
                summary = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(final_metrics).items()}
                print(f"[Eval] Final metrics: {summary}")
            except Exception:
                pass
            # Override last eval metrics with the final evaluation metrics
            try:
                last_eval_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(final_metrics).items()}
            except Exception:
                pass
    except Exception as e:
        print(f"[Warning] Final eval attempt skipped: {e}")

    # Merge/save if requested
    # Note: YAML null should disable merging entirely. Avoid coercing None to "None".
    _merge_raw = train_cfg.get("merge_output_dir", None)
    merge_dir: Optional[str] = None
    if isinstance(_merge_raw, str):
        _tmp = _merge_raw.strip()
        if _tmp:
            merge_dir = _tmp
    elif _merge_raw:
        # Allow non-string truthy values, but coerce explicitly
        merge_dir = str(_merge_raw)

    if swipe_name and merge_dir:
        merge_dir = os.path.join(merge_dir, swipe_name)

    if merge_dir:
        print("Finished training, merging model...")
        os.makedirs(merge_dir, exist_ok=True)
        # Merge once, then move to CPU and set dtype
        merged_model = model.merge_and_unload()
        try:
            merged_model = merged_model.to("cpu", dtype=torch.float16)
        except Exception:
            try:
                merged_model = merged_model.to("cpu")
            except Exception:
                pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        merged_model.save_pretrained(merge_dir, max_shard_size="3GB")
        tokenizer.save_pretrained(merge_dir)

        # Persist last eval metrics alongside the merged model
        try:
            metrics_path = os.path.join(merge_dir, "last_eval_metrics.json")
            payload: Dict[str, Any] = {}
            if last_eval_metrics is not None:
                payload = {str(k): (float(v) if isinstance(v, (int, float)) else v) for k, v in last_eval_metrics.items()}
            else:
                payload = {"note": "No evaluation metrics were available."}
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[Eval] Wrote last eval metrics to: {metrics_path}")

            with open(os.path.join(merge_dir, "hparams.json"), "w", encoding="utf-8") as f:
                json.dump(hparams, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Warning] Could not write last eval metrics JSON: {e}")

        print("Model merged and saved.")

        # Cleanup: remove optimizer.pt files from checkpoints to save space
        try:
            checkpoint_root = output_dir
            removed = 0
            if os.path.isdir(checkpoint_root):
                for root, dirs, files in os.walk(checkpoint_root):
                    # Only act on HF Trainer checkpoint folders
                    if os.path.basename(root).startswith("checkpoint"):
                        opt_path = os.path.join(root, "optimizer.pt")
                        if os.path.isfile(opt_path):
                            try:
                                os.remove(opt_path)
                                removed += 1
                            except Exception as e:
                                print(f"[Cleanup] Failed to remove {opt_path}: {e}")
            print(f"[Cleanup] Removed optimizer.pt from {removed} checkpoint(s) under '{checkpoint_root}'.")
        except Exception as e:
            print(f"[Cleanup] Skipped optimizer cleanup due to error: {e}")
    else:
        # Explicitly confirm that merging was skipped when merge_output_dir is null/empty.
        print("[Merge] Skipped: training.training.merge_output_dir is null/empty.")


if __name__ == "__main__":
    main()
