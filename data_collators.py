from typing import Union, Any, Dict, List
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorForLastCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template: Union[str, list[int]], *args, mlm: bool = False, ignore_index: int = -100, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.response_token_ids = (
            self.tokenizer.encode(response_template, add_special_tokens=False)
            if isinstance(response_template, str)
            else response_template
        )
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
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
