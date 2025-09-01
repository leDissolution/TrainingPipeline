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

    def torch_call(self, examples: List[Any]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
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

        # Optionally forward a per-token mask from dataset examples
        if self.fork_mask_key:
            try:
                import numpy as np  # optional, only if dataset stores numpy arrays
            except Exception:
                np = None  # type: ignore

            B, S = input_ids.shape
            # Start with ones if auto_full_mask, else zeros and try to fill from examples
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
