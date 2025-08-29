from transformers.trainer_callback import TrainerCallback
from typing import List, Tuple, Optional, Any
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
