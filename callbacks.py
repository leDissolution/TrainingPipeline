from transformers.trainer_callback import TrainerCallback
from typing import List, Tuple, Optional, Any, Dict, DefaultDict, Iterable, Callable
import os
import json

class LoraDropoutAnneal(TrainerCallback):
    """
    Linearly anneals LoRA-dropout from 'start_p' to 'end_p'
    over 'anneal_steps' optimisation steps.
    """

    def __init__(self, start_p: float = 0.0,
                       end_p:   float = 0.25,
                       anneal_steps: int = 1000):
        self.start_p = start_p
        self.end_p   = end_p
        self.anneal_steps = anneal_steps

    def _current_p(self, step: int) -> float:
        if step >= self.anneal_steps:
            return self.end_p
        frac = step / self.anneal_steps
        return self.start_p + frac * (self.end_p - self.start_p)

    def on_step_begin(self, args, state, control, **kwargs):
        p = self._current_p(state.global_step)
        model = kwargs["model"]

        for module in model.modules():
            if hasattr(module, "lora_dropout"):
                module.lora_dropout.default.p = p

class HeadDropAnneal(TrainerCallback):
    def __init__(self, collator, start_p, end_p, anneal_steps):
        super().__init__()
        self.collator = collator
        self.start, self.end, self.total = start_p, end_p, anneal_steps

    def on_step_begin(self, args, state, control, **kwargs):
        frac = min(state.global_step / self.total, 1.0)
        p = self.start + frac * (self.end - self.start)
        self.collator.head_p = p

class HeadDropEvalControl(TrainerCallback):
    def __init__(self, collator):
        super().__init__()
        self.collator = collator

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(state, "is_evaluating", False):
            self.collator.head_p = 0.0
    # No return to comply with TrainerCallback's expected signature (returns None)

import numpy as np
import torch
import re
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

class MetricCalculator:
    def __init__(self, tokenizer, ignore_index: int = -100,
                 expected_values: Optional[List[str]] = None,
                 target_attrs: Optional[List[str]] = None,
                 example_ids: Optional[List[Any]] = None):
        self.tok = tokenizer
        self.ignore_index = ignore_index
        self.noop_id = tokenizer.convert_tokens_to_ids("!!no_change!!")
        print(f"Using noop_id: {self.noop_id} ({tokenizer.convert_ids_to_tokens(self.noop_id)})")
        # Optional, if provided we'll use these references for text matching metrics
        self.expected_values: Optional[List[str]] = list(expected_values) if expected_values is not None else None
        # Optional per-example attribute for grouped metrics (e.g., example['target']['attr'])
        self.target_attrs: Optional[List[str]] = list(target_attrs) if target_attrs is not None else None
        # Optional ids for per-example logging
        self.example_ids: Optional[List[Any]] = list(example_ids) if example_ids is not None else None
        # Last per-example success flags (populated on every compute_metrics call)
        self.last_per_example_success: Optional[List[bool]] = None
        # Text helpers
        self.closing_suffix = '" />'
        try:
            self.noop_str = tokenizer.decode([self.noop_id], skip_special_tokens=True).strip() if self.noop_id is not None and self.noop_id >= 0 else ""
        except Exception:
            self.noop_str = "!!no_change!!"

    def keep_argmax(self, logits, labels):
        # logits: (B, S, V)
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # 1)  Decode only the completion part of every sequence - vectorised #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _decode_completions(
        self,
        pred_ids: torch.Tensor,          # (B, S)
        label_ids: torch.Tensor,         # (B, S)
    ) -> Tuple[List[str], List[str]]:

        mask = label_ids.ne(self.ignore_index)          # (B, S) bool
        pred_seqs  = [p[m].tolist() for p, m in zip(pred_ids, mask)]
        label_seqs = [l[m].tolist() for l, m in zip(label_ids, mask)]

        preds = self.tok.batch_decode(
            pred_seqs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        refs = self.tok.batch_decode(
            label_seqs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return [p.strip() for p in preds], [r.strip() for r in refs]

    # ------------------------------------------------------------ #
    # 2)  Metric entry point used by HF Trainer / accelerate etc.  #
    # ------------------------------------------------------------ #
    @torch.no_grad()
    def compute_metrics(self, eval_pred):
        # `eval_pred` is usually a tuple of (predictions, label_ids)
        # Keep backward compatibility with both tuple and EvalPrediction-like objects
        if isinstance(eval_pred, tuple) or isinstance(eval_pred, list):
            pred_ids, label_ids = map(torch.as_tensor, eval_pred)
        else:
            # EvalPrediction-style with attributes
            pred_ids = torch.as_tensor(getattr(eval_pred, "predictions"))
            label_ids = torch.as_tensor(getattr(eval_pred, "label_ids"))

        # 1) shift (the trainer already shifted labels for
        #    causal-LMs, so usually this is no longer necessary)
        # If predictions are logits (B, S, V), convert to token ids first
        if pred_ids.dim() == 3:
            pred_ids = torch.argmax(pred_ids, dim=-1)
        pred_ids = pred_ids[:, :-1]
        label_ids = label_ids[:, 1:]

        # noop metrics computed at token-level before decoding
        with torch.no_grad():
            valid_mask = label_ids.ne(self.ignore_index)
            if self.noop_id is not None and self.noop_id >= 0:
                is_noop_label = label_ids.eq(self.noop_id) & valid_mask
                is_noop_pred  = pred_ids.eq(self.noop_id) & valid_mask

                total_noop_labels = int(is_noop_label.sum().item())
                correct_noops = int((is_noop_label & is_noop_pred).sum().item())
                noop_acc_rate = (correct_noops / total_noop_labels) if total_noop_labels > 0 else 0.0

                non_noop_mask = valid_mask & (~is_noop_label)
                total_non_noop = int(non_noop_mask.sum().item())
                overused_noops = int((is_noop_pred & non_noop_mask).sum().item())
                noop_overuse = (overused_noops / total_non_noop) if total_non_noop > 0 else 0.0
            else:
                noop_acc_rate = 0.0
                noop_overuse = 0.0

        # 2) extract completion token id sequences for per-example checks
        mask = label_ids.ne(self.ignore_index)  # (B, S) bool
        pred_seqs_ids  = [p[m].tolist() for p, m in zip(pred_ids, mask)]
        label_seqs_ids = [l[m].tolist() for l, m in zip(label_ids, mask)]

        # 2a) decode predicted completion strings
        preds = self.tok.batch_decode(pred_seqs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds = [p.strip() for p in preds]

        # 2b) choose references for text matching
        # Prefer dataset-provided expected values when available; otherwise fall back to label decoding
        if self.expected_values is not None:
            # Trim or pad expected_values to match preds length (safety)
            if len(self.expected_values) < len(preds):
                refs = self.expected_values + [""] * (len(preds) - len(self.expected_values))
            else:
                refs = self.expected_values[: len(preds)]
            # Append closing suffix that isn't baked into expected_value
            refs = [(r or "") + self.closing_suffix for r in refs]
        else:
            # Fall back to decoding labels to get refs
            _, refs = self._decode_completions(pred_ids, label_ids)

        # 3) helpers for text metrics
        def partial_score(p, r, separators):
            def split_multi(s, seps):
                for sep in seps:
                    s = s.replace(sep, ',')
                return [x.strip().lower() for x in s.split(',')]
            p_parts = split_multi(p, separators)
            r_parts = split_multi(r, separators)
            return (sum(x in r_parts for x in p_parts) / len(r_parts)) if r_parts else 0.0

        def jaccard_score(p, r, separators):
            def split_multi(s, seps):
                for sep in seps:
                    s = s.replace(sep, ',')
                return {seg.strip().lower() for seg in s.split(',') if seg.strip()}
            P = split_multi(p, separators)
            R = split_multi(r, separators)
            if not P and not R:
                return 1.0
            return len(P & R) / len(P | R)

        def normalize_span(text: str) -> str:
            text = text.lower().replace('"', '').strip()
            text = re.sub(r'[,\s]+', ' ', text)
            return text

    # 3a) acceptance rule: if first completion label is noop, accept either
    #      a) prediction equals expected ref+closing suffix, or
    #      b) prediction equals noop token + closing suffix
        def is_label_first_noop(seq_ids: List[int]) -> bool:
            return len(seq_ids) > 0 and seq_ids[0] == self.noop_id

        def eq_casefold(a: str, b: str) -> bool:
            return (a or "").strip().lower() == (b or "").strip().lower()

        exact_vals = []
        partial_vals = []
        jaccard_vals = []
        jaccard_word_vals = []
        per_example_success: List[bool] = []

        noop_with_closing = (self.noop_str + self.closing_suffix).strip()

        for p_text, r_text, p_ids, l_ids in zip(preds, refs, pred_seqs_ids, label_seqs_ids):
            if is_label_first_noop(l_ids):
                # Accept either expected_value+closing or noop+closing as correct
                if eq_casefold(p_text, r_text) or eq_casefold(p_text, noop_with_closing):
                    exact_vals.append(1.0)
                    partial_vals.append(1.0)
                    jaccard_vals.append(1.0)
                    jaccard_word_vals.append(1.0)
                    per_example_success.append(True)
                    continue
            # normal comparisons
            is_exact = bool(eq_casefold(p_text, r_text))
            exact_vals.append(1.0 if is_exact else 0.0)
            partial_vals.append(partial_score(p_text, r_text, [',', ';']))
            jaccard_vals.append(jaccard_score(p_text, r_text, [',', ';']))
            jaccard_word_vals.append(jaccard_score(normalize_span(p_text), normalize_span(r_text), [' ']))
            per_example_success.append(is_exact)

        exact_match = float(np.mean(exact_vals)) if exact_vals else 0.0
        partial_match = float(np.mean(partial_vals)) if partial_vals else 0.0
        jaccard_match = float(np.mean(jaccard_vals)) if jaccard_vals else 0.0
        jaccard_word_match = float(np.mean(jaccard_word_vals)) if jaccard_word_vals else 0.0

        # 4) optional breakdown by target attr
        metrics = {
            "exact_match": exact_match,
            "partial_match": partial_match,
            "jaccard_match": jaccard_match,
            "jaccard_word_match": jaccard_word_match,
            "noop_acc_rate": noop_acc_rate,
            "noop_overuse": noop_overuse,
        }

        # Stash per-example success for external callbacks to persist
        try:
            self.last_per_example_success = list(per_example_success)
        except Exception:
            self.last_per_example_success = None

        # Grouped exact match by attr if provided
        if self.target_attrs is not None:
            attrs = self.target_attrs
            # Align length to preds
            if len(attrs) < len(preds):
                attrs = attrs + [""] * (len(preds) - len(attrs))
            else:
                attrs = attrs[: len(preds)]

            # correctness per example
            correct_flags = []
            for p_text, r_text, p_ids, l_ids in zip(preds, refs, pred_seqs_ids, label_seqs_ids):
                if is_label_first_noop(l_ids) and (eq_casefold(p_text, r_text) or eq_casefold(p_text, noop_with_closing)):
                    correct_flags.append(1.0)
                else:
                    correct_flags.append(1.0 if eq_casefold(p_text, r_text) else 0.0)
            counts = {}
            sums = {}
            for a, c in zip(attrs, correct_flags):
                if not a:
                    continue
                key = re.sub(r"[^0-9a-zA-Z_.-]", "_", str(a).lower())
                counts[key] = counts.get(key, 0) + 1
                sums[key] = sums.get(key, 0.0) + c
            for key in counts:
                metrics[f"e.{key}"] = float(sums[key] / counts[key])

        return metrics


class PerExampleEvalLogger(TrainerCallback):
    """Writes per-example success flags to JSONL per evaluation.

    Each line: {"id": <dataset id>, "success": <bool>}.
    Files are stored under `<output_dir>/per_example_eval/eval_step_<global_step>.jsonl`.
    """

    def __init__(self, metric_calculator: MetricCalculator, output_dir: str) -> None:
        super().__init__()
        self.metric_calculator = metric_calculator
        self.base_dir = os.path.join(str(output_dir), "per_example_eval")
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception:
            pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        try:
            flags = getattr(self.metric_calculator, "last_per_example_success", None)
            ids = getattr(self.metric_calculator, "example_ids", None)
            if not flags or not isinstance(flags, list):
                return
            # Align ids length if available
            if isinstance(ids, list) and len(ids) >= len(flags):
                ids_use = ids[: len(flags)]
            else:
                # Synthesize sequential indices if ids missing
                ids_use = list(range(len(flags)))

            step = int(getattr(state, "global_step", 0))
            path = os.path.join(self.base_dir, f"eval_step_{step}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i, ok in zip(ids_use, flags):
                    rec = {"id": i, "success": bool(ok)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Non-fatal: avoid breaking training due to logging issues
            return


class GradTensorBoardLogger(TrainerCallback):
    """Logs gradient stats per layer and module kind (mlp vs self_attn) to TensorBoard.

        Implementation details:
        - Registers autograd hooks on trainable parameters once at train start.
        - On each step, aggregates sum of squares and abs sums per (layer_index, kind).
            Kind is determined from parameter name containing ".mlp." or ".self_attn.".
        - On logging steps (control.should_log), writes scalars to TB under tags like:
            grads/layer_{i}/self_attn/l2
            grads/layer_{i}/self_attn/mean_abs
            grads/layer_{i}/mlp/l2
            grads/layer_{i}/mlp/mean_abs

        Logging location behavior:
        - By default, this callback writes into the SAME TensorBoard run directory as the
            Trainer's built-in TensorBoardCallback (args.logging_dir), using a distinct
            filename suffix (e.g., "events...grads"). This ensures both logs appear together
            in TensorBoard without overwriting each other.
        - You can override the directory via `logging_dir` or disable reuse of
            `args.logging_dir` by setting `use_trainer_logging_dir=False`.

        Notes:
        - Overhead: scans each gradient tensor once via simple reductions.
        - Safe with gradient accumulation: hooks see every backward; we reset
            accumulators at step begin so stats reflect the current optimizer step.
        """

    # Class-level annotations for static analysis
    _writer: Optional[SummaryWriter]
    _acc: DefaultDict[Tuple[int, str], Dict[str, float]]
    _handles: List[Any]
    _seen_step: Optional[int]

    def __init__(
        self,
        logging_dir: Optional[str] = None,
        include_kinds: Iterable[str] = ("self_attn", "mlp"),
        tag_prefix: str = "grads",
        use_trainer_logging_dir: bool = True,
        filename_suffix: str = ".grads",
    ) -> None:
        super().__init__()
        # May be None; resolved on train begin using args.logging_dir if requested
        self.logging_dir = str(logging_dir) if logging_dir is not None else None
        self.include_kinds = {str(k) for k in include_kinds}
        self.tag_prefix = str(tag_prefix).strip() or "grads"
        self.use_trainer_logging_dir = bool(use_trainer_logging_dir)
        self.filename_suffix = str(filename_suffix)
        self._writer = None  # type: ignore[assignment]
        # (layer_idx:int, kind:str) -> {sq_sum:float, abs_sum:float, count:float}
        self._acc = defaultdict(lambda: {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0.0})  # type: ignore[assignment]
        self._handles = []
        self._seen_step = None

    def _parse_param_group(self, name: str) -> Optional[Tuple[int, str]]:
        # Find layer index
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if not m:
            return None
        layer_idx = int(m.group(1))
        kind = None
        if ".self_attn." in name:
            kind = "self_attn"
        elif ".mlp." in name:
            kind = "mlp"
        if kind is None or kind not in self.include_kinds:
            return None
        return (layer_idx, kind)

    def _register_hooks(self, model) -> None:
        # Register on all named parameters once
        iterator = model.named_parameters() if hasattr(model, "named_parameters") else []
        for name, p in iterator:
            if not isinstance(p, torch.nn.Parameter) or not p.requires_grad:
                continue
            key = self._parse_param_group(name)
            if key is None:
                continue

            def _make_hook(k: Tuple[int, str]):
                def _hook(grad: torch.Tensor):
                    try:
                        if grad is None or grad.numel() == 0:
                            return grad
                        ds = self._acc[k]
                        # Use CPU item extractions to avoid storing tensors
                        # Sum of squares and abs sum across all elements
                        ds["sq_sum"] += float(torch.sum(grad.detach() * grad.detach()).item())
                        ds["abs_sum"] += float(torch.sum(grad.detach().abs()).item())
                        ds["count"] += float(grad.numel())
                    except Exception:
                        pass
                    return grad
                return _hook

            h = p.register_hook(_make_hook(key))
            self._handles.append(h)

    # ------------- Callback events -------------
    def on_train_begin(self, args, state, control, **kwargs):
        # Create writer lazily and hook parameters
        try:
            if self._writer is None:
                # Resolve effective log directory
                effective_dir = None
                if self.use_trainer_logging_dir and hasattr(args, "logging_dir") and getattr(args, "logging_dir"):
                    effective_dir = str(getattr(args, "logging_dir"))
                elif self.logging_dir is not None:
                    effective_dir = self.logging_dir
                else:
                    # Fallback to common default: <output_dir>/runs
                    base = getattr(args, "output_dir", None)
                    if base:
                        effective_dir = os.path.join(str(base), "runs")
                if effective_dir is None:
                    # Final safety: current working directory "runs"
                    effective_dir = os.path.join(os.getcwd(), "runs")
                os.makedirs(effective_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=effective_dir, filename_suffix=self.filename_suffix)
        except Exception:
            self._writer = None
        model = kwargs.get("model")
        if model is not None and not self._handles:
            try:
                self._register_hooks(model)
            except Exception:
                pass

    def on_step_begin(self, args, state, control, **kwargs):
        # Reset accumulators when a new optimizer step begins
        step = int(getattr(state, "global_step", 0))
        if self._seen_step is None or self._seen_step != step:
            self._acc.clear()
            self._seen_step = step

    def on_step_end(self, args, state, control, **kwargs):
        if not control.should_log:
            return
        if self._writer is None:
            return
        step = int(getattr(state, "global_step", 0))
        # Aggregate and write
        try:
            # Optional: global per-kind aggregator across layers
            global_kind_acc: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0.0})
            for (layer_idx, kind), stats in list(self._acc.items()):
                count = max(1.0, float(stats.get("count", 0.0)))
                l2 = (float(stats.get("sq_sum", 0.0)) ** 0.5)
                mean_abs = float(stats.get("abs_sum", 0.0)) / count
                # Per-layer
                base = f"{self.tag_prefix}/layer_{layer_idx}/{kind}"
                try:
                    self._writer.add_scalar(f"{base}/l2", l2, step)
                    self._writer.add_scalar(f"{base}/mean_abs", mean_abs, step)
                except Exception:
                    pass
                # Accumulate global per-kind
                g = global_kind_acc[kind]
                g["sq_sum"] += float(stats.get("sq_sum", 0.0))
                g["abs_sum"] += float(stats.get("abs_sum", 0.0))
                g["count"] += float(stats.get("count", 0.0))

            # Global per-kind
            for kind, g in global_kind_acc.items():
                cnt = max(1.0, float(g.get("count", 0.0)))
                l2 = (float(g.get("sq_sum", 0.0)) ** 0.5)
                mean_abs = float(g.get("abs_sum", 0.0)) / cnt
                base = f"{self.tag_prefix}/all_layers/{kind}"
                try:
                    self._writer.add_scalar(f"{base}/l2", l2, step)
                    self._writer.add_scalar(f"{base}/mean_abs", mean_abs, step)
                except Exception:
                    pass
            try:
                self._writer.flush()
            except Exception:
                pass
        except Exception:
            # Swallow to avoid disrupting training
            return

    def on_train_end(self, args, state, control, **kwargs):
        # Remove hooks and close writer
        try:
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._handles.clear()
        except Exception:
            pass
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception:
            pass


class ForkAlphaScheduler(TrainerCallback):
    """Schedules trainer._fork_alpha over steps and logs it to TensorBoard.

    Config schema (YAML under loss.fork_alpha_schedule):
      type: "linear" | "cosine" | "constant"
      args:
        start_rate: float   # multiplier of base alpha at schedule start
        end_rate: float     # multiplier at schedule end
        start_step: float   # fraction of total steps where schedule begins (0..1)
        end_step: float     # fraction of total steps where schedule ends (0..1)

    Notes:
    - For type == constant, args are ignored; alpha stays at base_alpha.
    - Outside [start_step, end_step], alpha clamps to start_rate*base or end_rate*base.
    - Writes a scalar "fork_alpha" at logging steps, alongside loss.
    """

    def __init__(
        self,
        base_alpha: float,
        sched_type: str,
        args: Optional[Dict[str, float]] = None,
        on_update: Optional[Callable[[float], None]] = None,
        logging_dir: Optional[str] = None,
        use_trainer_logging_dir: bool = True,
        filename_suffix: str = ".fork_alpha",
    ) -> None:
        super().__init__()
        self.base_alpha = float(base_alpha)
        self.sched_type = str(sched_type or "constant").lower()
        self.args = dict(args or {})
        self._writer: Optional[SummaryWriter] = None
        self._total_steps: int = 0
        self._current_alpha: float = float(base_alpha)
        self._on_update = on_update
        self.logging_dir = str(logging_dir) if logging_dir is not None else None
        self.use_trainer_logging_dir = bool(use_trainer_logging_dir)
        self.filename_suffix = str(filename_suffix)

    # ---- internals ----
    def _resolve_total_steps(self, args, state) -> int:
        try:
            ms = int(getattr(state, "max_steps", 0) or 0)
            if ms and ms > 0:
                return ms
        except Exception:
            pass
        try:
            ms = int(getattr(args, "max_steps", 0) or 0)
            if ms and ms > 0:
                return ms
        except Exception:
            pass
        return 1

    def _calc_rate(self, step: int) -> float:
        if self.sched_type == "constant":
            return 1.0

        # Pull with defaults
        start_rate = float(self.args.get("start_rate", 1.0))
        end_rate = float(self.args.get("end_rate", 1.0))
        start_frac = float(self.args.get("start_step", 0.0))
        end_frac = float(self.args.get("end_step", 1.0))

        # Guard rails
        start_frac = max(0.0, min(1.0, start_frac))
        end_frac = max(0.0, min(1.0, end_frac))
        if end_frac < start_frac:
            start_frac, end_frac = end_frac, start_frac

        s0 = int(round(start_frac * self._total_steps))
        s1 = int(round(end_frac * self._total_steps))
        s1 = max(s1, s0 + 1)

        if step <= s0:
            return start_rate
        if step >= s1:
            return end_rate

        # Progress in [0,1]
        t = (step - s0) / max(1, (s1 - s0))
        if self.sched_type == "linear":
            return start_rate + t * (end_rate - start_rate)
        elif self.sched_type == "cosine":
            # Cosine interpolation between start_rate and end_rate
            # rate = start + (1 - cos(pi*t))/2 * (end-start)
            try:
                import math
                return start_rate + 0.5 * (1.0 - math.cos(math.pi * t)) * (end_rate - start_rate)
            except Exception:
                return start_rate + t * (end_rate - start_rate)
        else:
            # Fallback to linear if unknown type
            return start_rate + t * (end_rate - start_rate)

    def _update_alpha(self, step: int) -> None:
        rate = self._calc_rate(step)
        self._current_alpha = float(self.base_alpha * rate)
        # Propagate to owner via callback
        try:
            if self._on_update is not None:
                self._on_update(self._current_alpha)
        except Exception:
            pass

    # ---- callbacks ----
    def on_train_begin(self, args, state, control, **kwargs):
        # Setup writer
        try:
            if self._writer is None:
                effective_dir = None
                if self.use_trainer_logging_dir and hasattr(args, "logging_dir") and getattr(args, "logging_dir"):
                    effective_dir = str(getattr(args, "logging_dir"))
                elif self.logging_dir is not None:
                    effective_dir = self.logging_dir
                else:
                    base = getattr(args, "output_dir", None)
                    if base:
                        effective_dir = os.path.join(str(base), "runs")
                if effective_dir is None:
                    effective_dir = os.path.join(os.getcwd(), "runs")
                os.makedirs(effective_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=effective_dir, filename_suffix=self.filename_suffix)
        except Exception:
            self._writer = None

        # Resolve total steps and prime current alpha
        self._total_steps = self._resolve_total_steps(args, state)
        self._update_alpha(int(getattr(state, "global_step", 0) or 0))

    def on_step_begin(self, args, state, control, **kwargs):
        self._update_alpha(int(getattr(state, "global_step", 0) or 0))

    def on_step_end(self, args, state, control, **kwargs):
        if not control.should_log:
            return
        if self._writer is None:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        try:
            self._writer.add_scalar("train/fork_alpha", float(self._current_alpha), step)
            self._writer.flush()
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception:
            pass
