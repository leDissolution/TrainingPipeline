import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class TensorDiff:
	name: str
	status: str  # ok | missing_a | missing_b | shape_mismatch
	shape_a: Optional[Tuple[int, ...]]
	shape_b: Optional[Tuple[int, ...]]
	l2: float = 0.0
	mean_abs: float = 0.0
	max_abs: float = 0.0
	rel_l2: float = 0.0
	numel: int = 0
	changed: bool = False
	embed_token_rows: Optional[List[Tuple[int, Optional[str], float, float, float, float]]] = None


def _make_token_decoder(path: str, revision: Optional[str], trust_remote_code: bool) -> Optional[Callable[[int], Optional[str]]]:
	try:
		tokenizer = AutoTokenizer.from_pretrained(
			path,
			revision=revision,
			trust_remote_code=trust_remote_code,
		)
		vocab_size = tokenizer.vocab_size

		def decode(idx: int) -> Optional[str]:
			if idx < 0 or idx >= vocab_size:
				return None
			try:
				tok = tokenizer.convert_ids_to_tokens([idx])
				return tok[0] if tok else None
			except Exception:
				return None

		return decode
	except Exception:
		return None


def _dtype_from_arg(arg: str) -> torch.dtype:
	arg = arg.lower()
	if arg in {"fp16", "float16", "half"}:
		return torch.float16
	if arg in {"bf16", "bfloat16"}:
		return torch.bfloat16
	if arg in {"fp32", "float32", "float"}:
		return torch.float32
	raise ValueError(f"Unsupported dtype: {arg}")


def _load_state_dict(path: str, revision: Optional[str], dtype: torch.dtype, trust_remote_code: bool) -> Dict[str, torch.Tensor]:
	model = AutoModel.from_pretrained(
		path,
		revision=revision,
		torch_dtype=dtype,
		device_map="cpu",
		low_cpu_mem_usage=True,
		trust_remote_code=trust_remote_code,
	)
	model.to("cpu")
	state = {k: v.to(torch.float32).cpu() for k, v in model.state_dict().items()}
	return state


def _compare_state_dicts(
	state_a: Dict[str, torch.Tensor],
	state_b: Dict[str, torch.Tensor],
	rtol: float,
	atol: float,
	filter_substr: Optional[str],
	verbose_embeddings: bool,
	embed_top_k: int,
	token_decoder: Optional[Callable[[int], Optional[str]]],
) -> List[TensorDiff]:
	all_keys = sorted(set(state_a.keys()) | set(state_b.keys()))
	diffs: List[TensorDiff] = []
	for name in all_keys:
		if filter_substr and filter_substr not in name:
			continue
		ta = state_a.get(name)
		tb = state_b.get(name)
		if ta is None:
			diffs.append(TensorDiff(name=name, status="missing_a", shape_a=None, shape_b=tuple(tb.shape)))
			continue
		if tb is None:
			diffs.append(TensorDiff(name=name, status="missing_b", shape_a=tuple(ta.shape), shape_b=None))
			continue
		if ta.shape != tb.shape:
			# Special-case embeddings: compare overlapping rows if only vocab size differs.
			if (
				len(ta.shape) == 2
				and len(tb.shape) == 2
				and ta.shape[1] == tb.shape[1]
				and "embed_tokens" in name
			):
				rows = min(ta.shape[0], tb.shape[0])
				ta_sub = ta[:rows]
				tb_sub = tb[:rows]
				delta = (ta_sub - tb_sub).float()
				l2 = delta.norm().item()
				mean_abs = delta.abs().mean().item()
				max_abs = delta.abs().max().item()
				base = ta_sub.norm().item()
				rel_l2 = l2 / (base + 1e-12)
				changed = not torch.allclose(ta_sub, tb_sub, rtol=rtol, atol=atol)
				embed_rows = None
				if verbose_embeddings:
					# Compute per-token metrics and retain top-k rows by L2 diff.
					row_l2 = delta.norm(dim=1)
					row_mean_abs = delta.abs().mean(dim=1)
					row_max_abs = delta.abs().max(dim=1).values
					row_base = ta_sub.norm(dim=1)
					row_rel_l2 = row_l2 / (row_base + 1e-12)
					k = min(embed_top_k, delta.shape[0])
					if k > 0:
						top_vals, top_idx = torch.topk(row_l2, k)
						embed_rows = []
						for idx, val in zip(top_idx, top_vals):
							int_idx = int(idx.item())
							embed_rows.append(
								(
									int_idx,
									token_decoder(int_idx) if token_decoder else None,
									float(val.item()),
									float(row_mean_abs[idx].item()),
									float(row_max_abs[idx].item()),
									float(row_rel_l2[idx].item()),
								)
							)
				diffs.append(
					TensorDiff(
						name=name,
						status="ok_overlap",
						shape_a=tuple(ta.shape),
						shape_b=tuple(tb.shape),
						l2=l2,
						mean_abs=mean_abs,
						max_abs=max_abs,
						rel_l2=rel_l2,
						numel=ta_sub.numel(),
						changed=changed,
						embed_token_rows=embed_rows,
					)
				)
				continue
			# Generic mismatch
			diffs.append(
				TensorDiff(
					name=name,
					status="shape_mismatch",
					shape_a=tuple(ta.shape),
					shape_b=tuple(tb.shape),
				)
			)
			continue

		delta = (ta - tb).float()
		l2 = delta.norm().item()
		mean_abs = delta.abs().mean().item()
		max_abs = delta.abs().max().item()
		base = ta.norm().item()
		rel_l2 = l2 / (base + 1e-12)
		changed = not torch.allclose(ta, tb, rtol=rtol, atol=atol)
		embed_rows = None
		if verbose_embeddings and len(delta.shape) == 2 and "embed_tokens" in name:
			row_l2 = delta.norm(dim=1)
			row_mean_abs = delta.abs().mean(dim=1)
			row_max_abs = delta.abs().max(dim=1).values
			row_base = ta.norm(dim=1)
			row_rel_l2 = row_l2 / (row_base + 1e-12)
			k = min(embed_top_k, delta.shape[0])
			if k > 0:
				top_vals, top_idx = torch.topk(row_l2, k)
				embed_rows = []
				for idx, val in zip(top_idx, top_vals):
					int_idx = int(idx.item())
					embed_rows.append(
						(
							int_idx,
							token_decoder(int_idx) if token_decoder else None,
							float(val.item()),
							float(row_mean_abs[idx].item()),
							float(row_max_abs[idx].item()),
							float(row_rel_l2[idx].item()),
						)
					)
		diffs.append(
			TensorDiff(
				name=name,
				status="ok",
				shape_a=tuple(ta.shape),
				shape_b=tuple(tb.shape),
				l2=l2,
				mean_abs=mean_abs,
				max_abs=max_abs,
				rel_l2=rel_l2,
				numel=ta.numel(),
				changed=changed,
				embed_token_rows=embed_rows,
			)
		)
	return diffs


def _aggregate_by_module(diffs: Iterable[TensorDiff]) -> Dict[str, Dict[str, float]]:
	agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {
		"sq_sum": 0.0,
		"abs_sum": 0.0,
		"count": 0,
		"max_abs": 0.0,
		"changed_tensors": 0,
	})
	for d in diffs:
		module = d.name.rsplit(".", 1)[0] if "." in d.name else "(root)"
		entry = agg[module]
		if d.status in {"ok", "ok_overlap"}:
			entry["sq_sum"] += d.l2 ** 2
			entry["abs_sum"] += d.mean_abs * d.numel
			entry["count"] += d.numel
			entry["max_abs"] = max(entry["max_abs"], d.max_abs)
			if d.changed:
				entry["changed_tensors"] += 1
		else:
			entry["changed_tensors"] += 1
	# finalize
	out: Dict[str, Dict[str, float]] = {}
	for module, vals in agg.items():
		count = max(vals["count"], 1)
		out[module] = {
			"l2": (vals["sq_sum"] ** 0.5),
			"mean_abs": vals["abs_sum"] / count,
			"max_abs": vals["max_abs"],
			"changed_tensors": vals["changed_tensors"],
		}
	return out


def _print_report(diffs: List[TensorDiff], top_k: int, verbose_embeddings: bool) -> None:
	missing_a = [d for d in diffs if d.status == "missing_a"]
	missing_b = [d for d in diffs if d.status == "missing_b"]
	shape_mismatch = [d for d in diffs if d.status == "shape_mismatch"]
	comparable = [d for d in diffs if d.status in {"ok", "ok_overlap"}]
	changed = [d for d in comparable if d.changed]

	total_params = sum(d.numel for d in comparable)
	changed_params = sum(d.numel for d in changed)

	print("==== Model Comparison ====")
	print(f"Compared tensors: {len(comparable)} | params: {total_params:,} | changed params: {changed_params:,}")
	if missing_a:
		print(f"Missing in A: {len(missing_a)} tensors")
	if missing_b:
		print(f"Missing in B: {len(missing_b)} tensors")
	if shape_mismatch:
		print(f"Shape mismatch: {len(shape_mismatch)} tensors")

	agg = _aggregate_by_module(diffs)
	ranked = sorted(agg.items(), key=lambda kv: kv[1]["l2"], reverse=True)
	print("\nTop modules by L2 diff:")
	for module, stats in ranked[:top_k]:
		print(
			f"- {module}: l2={stats['l2']:.4f}, mean_abs={stats['mean_abs']:.6f}, "
			f"max_abs={stats['max_abs']:.6f}, changed_tensors={stats['changed_tensors']}"
		)

	if changed:
		changed_sorted = sorted(changed, key=lambda d: d.l2, reverse=True)
		print("\nTop tensors by L2 diff:")
		for d in changed_sorted[:top_k]:
			print(
				f"- {d.name}: l2={d.l2:.4f}, rel_l2={d.rel_l2:.6f}, "
				f"mean_abs={d.mean_abs:.6f}, max_abs={d.max_abs:.6f}, shape={d.shape_a}"
			)

	if verbose_embeddings:
		embed_diffs = [d for d in changed if d.embed_token_rows]
		if embed_diffs:
			print("\nPer-token embedding changes:")
			for d in embed_diffs:
				print(f"- {d.name} (shape={d.shape_a}):")
				for idx, token, l2, mean_abs, max_abs, rel_l2 in d.embed_token_rows[:top_k]:
					tok_display = token if token is not None else "<unk>"
					print(
						f"    token {idx} ({tok_display}): l2={l2:.4f}, rel_l2={rel_l2:.6f}, "
						f"mean_abs={mean_abs:.6f}, max_abs={max_abs:.6f}"
					)

	if missing_a:
		print("\nTensors missing in A:")
		for d in missing_a[:top_k]:
			print(f"- {d.name} shape_b={d.shape_b}")

	if missing_b:
		print("\nTensors missing in B:")
		for d in missing_b[:top_k]:
			print(f"- {d.name} shape_a={d.shape_a}")

	if shape_mismatch:
		print("\nShape mismatches:")
		for d in shape_mismatch[:top_k]:
			print(f"- {d.name}: {d.shape_a} vs {d.shape_b}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare two HF checkpoints and report changed modules.")
	parser.add_argument("model_a", help="Path or repo id for checkpoint A")
	parser.add_argument("model_b", help="Path or repo id for checkpoint B")
	parser.add_argument("--revision-a", dest="revision_a", default=None, help="Optional revision for checkpoint A")
	parser.add_argument("--revision-b", dest="revision_b", default=None, help="Optional revision for checkpoint B")
	parser.add_argument("--dtype", default="float32", help="Load dtype: float32, bf16, or float16")
	parser.add_argument("--rtol", type=float, default=1e-6, help="rtol for torch.allclose")
	parser.add_argument("--atol", type=float, default=1e-8, help="atol for torch.allclose")
	parser.add_argument("--top-k", type=int, default=20, help="Number of rows to show in each section")
	parser.add_argument("--filter", dest="filter_substr", default=None, help="Only compare tensors whose names contain this substring")
	parser.add_argument("--verbose-embeddings", action="store_true", help="Show per-token stats for embedding tensors")
	parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading")
	args = parser.parse_args()

	dtype = _dtype_from_arg(args.dtype)
	state_a = _load_state_dict(args.model_a, args.revision_a, dtype, args.trust_remote_code)
	state_b = _load_state_dict(args.model_b, args.revision_b, dtype, args.trust_remote_code)
	token_decoder = _make_token_decoder(args.model_a, args.revision_a, args.trust_remote_code) if args.verbose_embeddings else None

	diffs = _compare_state_dicts(
		state_a,
		state_b,
		args.rtol,
		args.atol,
		args.filter_substr,
		args.verbose_embeddings,
		args.top_k,
		token_decoder,
	)
	_print_report(diffs, args.top_k, args.verbose_embeddings)


if __name__ == "__main__":
	main()
