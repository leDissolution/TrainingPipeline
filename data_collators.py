from typing import Union, Any, Dict, List
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling

class DataCollatorForLastCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        response_template: Union[str, list[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        fork_mask_key: Union[str, None] = None,
        auto_full_mask: bool = False,
        mask_generator: Union[dict, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, mlm=mlm, **kwargs)
        self.response_token_ids = (
            self.tokenizer.encode(response_template, add_special_tokens=False)
            if isinstance(response_template, str)
            else response_template
        )
        self.ignore_index = ignore_index
        self.fork_mask_key = fork_mask_key
        self.auto_full_mask = bool(auto_full_mask)
        # Optional mask generator config (e.g., {type: "after_special_token", args: {...}})
        self._mask_gen_cfg = mask_generator if isinstance(mask_generator, dict) else None
        # Preprocess patterns for after_special_token
        self._mg_type = None
        self._mg_patterns: List[List[int]] = []
        self._mg_include_special: bool = False
        if self._mask_gen_cfg and str(self._mask_gen_cfg.get("type")) == "after_special_token":
            self._mg_type = "after_special_token"
            args = self._mask_gen_cfg.get("args", {}) or {}
            toks = args.get("tokens", []) or []
            self._mg_include_special = bool(args.get("include_special_tokens", False))
            patterns: List[List[int]] = []
            for t in toks:
                try:
                    if isinstance(t, str):
                        enc = self.tokenizer.encode(t, add_special_tokens=False)
                    elif isinstance(t, (list, tuple)):
                        enc = [int(x) for x in t]
                    else:
                        enc = []
                except Exception:
                    enc = []
                if enc:
                    patterns.append(enc)
            self._mg_patterns = patterns

    def torch_call(self, examples: List[Any]) -> Dict[str, Any]:
        # Extract per-example IDs up front from the raw examples
        extracted_ids: List[Any] | None = None
        try:
            if isinstance(examples[0], dict):
                id_keys = ["id", "example_id", "example_ids", "ids", "idx", "index"]
                chosen = None
                for k in id_keys:
                    if any(isinstance(ex, dict) and (k in ex) for ex in examples):
                        chosen = k
                        break
                if chosen is not None:
                    def _to_py(v: Any) -> Any:
                        try:
                            import numpy as _np  # optional
                            if isinstance(v, (_np.integer,)):
                                return int(v)
                            if isinstance(v, (_np.floating,)):
                                return float(v)
                            if isinstance(v, (_np.bool_,)):
                                return bool(v)
                        except Exception:
                            pass
                        if torch.is_tensor(v):
                            try:
                                return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
                            except Exception:
                                return None
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            return v
                        try:
                            return str(v)
                        except Exception:
                            return None

                    tmp_ids: List[Any] = []
                    for ex in examples:
                        val = ex.get(chosen) if isinstance(ex, dict) else None
                        if isinstance(val, (list, tuple)):
                            val = val[0] if len(val) > 0 else None
                        tmp_ids.append(_to_py(val))
                    extracted_ids = tmp_ids
        except Exception:
            extracted_ids = None

        # Sanitize examples: keep only tensorizable model keys before padding to avoid errors on text fields (e.g., 'prompt')
        clean_examples = examples
        try:
            if isinstance(examples[0], dict):
                allowed = {"input_ids", "labels", "attention_mask", "special_tokens_mask", "token_type_ids", "position_ids"}
                clean_examples = []
                for ex in examples:
                    if not isinstance(ex, dict):
                        clean_examples.append(ex)
                        continue
                    kept: Dict[str, Any] = {}
                    for k, v in ex.items():
                        if k in allowed:
                            kept[k] = v
                    clean_examples.append(kept)
        except Exception:
            clean_examples = examples

        batch = super().torch_call(clean_examples)
        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        attn_mask = batch["attention_mask"]

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for i in range(len(examples)):
            seq       = input_ids[i].tolist()
            labels_i  = labels[i]

            positions = [
                idx
                for idx in range(len(seq) - len(self.response_token_ids) + 1)
                if seq[idx : idx + len(self.response_token_ids)]
                   == self.response_token_ids
            ]
            if not positions:
                labels_i[:] = self.ignore_index
                continue

            completion_start = positions[-1] + len(self.response_token_ids)

            labels_i[:completion_start] = self.ignore_index

            pad_positions = (attn_mask[i] == 0).nonzero(as_tuple=True)[0]
            labels_i[pad_positions] = self.ignore_index

            for pos in range(completion_start, len(seq)):
                if seq[pos] == eos_id or seq[pos] == pad_id:
                    seq[pos] = eos_id
                    labels_i[pos] = eos_id
                    break

        batch["labels"] = labels

        # Pipe per-example IDs into the batch for downstream logging
        try:
            if extracted_ids is not None:
                batch["example_ids"] = extracted_ids
            else:
                # Fallback: stable hash of input_ids per example
                try:
                    import hashlib
                    ex_ids2: List[Any] = []
                    for row in input_ids:
                        try:
                            arr = row.detach().cpu().numpy().tobytes()
                        except Exception:
                            arr = bytes(str(row.detach().cpu().tolist()), encoding="utf-8")
                        h = hashlib.blake2b(arr, digest_size=8).hexdigest()
                        ex_ids2.append(h)
                    batch["example_ids"] = ex_ids2
                except Exception:
                    pass
        except Exception:
            pass

        # Optionally forward a per-token mask from dataset examples
        if self.fork_mask_key:
            try:
                import numpy as np  # optional, only if dataset stores numpy arrays
            except Exception:
                np = None  # type: ignore

            B, S = input_ids.shape
            # Strategy precedence:
            # 1) If mask_generator configured, generate from it.
            # 2) Else if auto_full_mask, make full ones.
            # 3) Else, try to pull mask from examples.
            if self._mg_type == "after_special_token":
                mask_tensor = torch.zeros((B, S), dtype=torch.float32, device=input_ids.device)
                eos_id = self.tokenizer.eos_token_id
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                # Collect special ids if requested
                special_ids: List[int] = []
                if self._mg_include_special:
                    try:
                        special_ids = [int(x) for x in (getattr(self.tokenizer, "all_special_ids", []) or []) if x is not None]
                    except Exception:
                        special_ids = []
                for i in range(B):
                    seq = input_ids[i].tolist()
                    # Valid length from attention mask (assume trailing pads)
                    try:
                        valid_len = int(attn_mask[i].sum().item())
                    except Exception:
                        valid_len = len(seq)
                    valid_len = max(0, min(valid_len, S))
                    # 1) Mark configured patterns and the token right after them
                    for pat in self._mg_patterns:
                        L = len(pat)
                        if L == 0 or L > valid_len:
                            continue
                        for j in range(0, valid_len - L + 1):
                            if seq[j:j+L] == pat:
                                # mark the pattern tokens themselves
                                mask_tensor[i, j:j+L] = 1.0
                                # and the token right after the pattern
                                if j + L < valid_len:
                                    mask_tensor[i, j+L] = 1.0
                    # 2) Mark special tokens and the token right after
                    if special_ids:
                        for j in range(valid_len):
                            if int(seq[j]) in special_ids:
                                mask_tensor[i, j] = 1.0
                                if j + 1 < valid_len:
                                    mask_tensor[i, j+1] = 1.0
                    # 3) Always treat EOS as special (even if not in all_special_ids)
                    if eos_id is not None:
                        for j in range(valid_len):
                            if seq[j] == eos_id:
                                mask_tensor[i, j] = 1.0
                                if j + 1 < valid_len:
                                    mask_tensor[i, j+1] = 1.0
                    # 4) Optionally BOS
                    if bos_id is not None and valid_len > 0 and seq[0] == bos_id:
                        mask_tensor[i, 0] = 1.0
                        if valid_len > 1:
                            mask_tensor[i, 1] = 1.0
                batch[self.fork_mask_key] = mask_tensor
            else:
                # Default behavior
                mask_tensor = torch.ones((B, S), dtype=torch.float32, device=input_ids.device) if self.auto_full_mask else torch.zeros((B, S), dtype=torch.float32, device=input_ids.device)
                if not self.auto_full_mask:
                    for i, ex in enumerate(examples):
                        if isinstance(ex, dict) and self.fork_mask_key in ex:
                            raw = ex[self.fork_mask_key]
                            vals: List[int] | None = None
                            try:
                                if isinstance(raw, torch.Tensor):
                                    vals = raw.detach().cpu().to(torch.int64).view(-1).tolist()
                                elif np is not None and hasattr(raw, "shape"):
                                    # numpy array
                                    try:
                                        vals = [int(x) for x in raw.reshape(-1).tolist()]
                                    except Exception:
                                        vals = None
                                elif isinstance(raw, (list, tuple)):
                                    vals = [int(x) for x in raw]
                                else:
                                    vals = None
                            except Exception:
                                vals = None
                            if vals is not None and len(vals) > 0:
                                # Align length to sequence length
                                if len(vals) >= S:
                                    use = vals[:S]
                                else:
                                    use = vals + [0] * (S - len(vals))
                                mask_tensor[i] = torch.tensor(use, dtype=torch.float32, device=input_ids.device)
                batch[self.fork_mask_key] = mask_tensor

        return batch

class DataCollatorLastCompWithHeadDropout(DataCollatorForLastCompletionOnlyLM):
    def __init__(
        self,
        response_template: Union[str, List[int]],
        num_attention_heads: int,
        head_dropout_p: float = 0.1,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__(response_template, *args, mlm=mlm, ignore_index=ignore_index, **kwargs)
        self.num_heads = num_attention_heads
        self.head_p    = head_dropout_p

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        attn2d = batch["attention_mask"]        # shape (B, S)
        bsz, seq_len = attn2d.shape

        mask4d = attn2d[:, None, None, :].float()

        keep = torch.bernoulli(
            torch.full((bsz, self.num_heads, 1, 1),
                       1.0 - self.head_p,
                       device=mask4d.device)
        )

        batch["attention_mask"] = mask4d * keep

        return batch
