#!/usr/bin/env python3
"""Estimate activation-gradient direction sign-flip probabilities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_alignment_dimension import (
    FixedTokenDataset,
    GradientCollector,
    find_layers,
    load_records,
    parse_layer_indices,
)


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in spec.split(","):
        fields = part.strip().split(":")
        if len(fields) == 1:
            m = k = int(fields[0])
        elif len(fields) == 2:
            m, k = map(int, fields)
        else:
            raise ValueError(f"Invalid M:K pair: {part!r}")
        if m < 1 or k < 1:
            raise ValueError("M and K must be positive")
        if (m, k) not in pairs:
            pairs.append((m, k))
    if not pairs:
        raise ValueError("No M:K pairs selected")
    return pairs


def build_splits(
    train_size: int,
    validation_size: int,
    pairs: list[tuple[int, int]],
    repetitions: int,
    seed: int,
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Create reproducible, row-wise non-overlapping M/K sample indices."""
    splits: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for m, k in pairs:
        if m > train_size or k > validation_size:
            raise ValueError(
                f"M={m} requires train pool >= {m} and K={k} requires validation pool >= {k}; "
                f"got {train_size} and {validation_size}"
            )
        train_indices = torch.rand((repetitions, train_size), generator=generator).argsort(dim=1)[:, :m]
        validation_indices = (
            torch.rand((repetitions, validation_size), generator=generator).argsort(dim=1)[:, :k]
        )
        splits[(m, k)] = (train_indices, validation_indices)
    return splits


def summarize_inner_products(
    q: torch.Tensor,
    split_indices: tuple[torch.Tensor, torch.Tensor],
    validation_q: torch.Tensor,
    m: int,
    k: int,
    trial_batch_size: int,
    save_inner_products: bool,
) -> dict[str, Any]:
    q = q.to(device="cpu", dtype=torch.float64)
    validation_q = validation_q.to(device="cpu", dtype=torch.float64)
    train_indices, validation_indices = split_indices
    values: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    for start in range(0, train_indices.shape[0], trial_batch_size):
        train_batch = train_indices[start : start + trial_batch_size]
        validation_batch = validation_indices[start : start + trial_batch_size]
        mu_m = q[train_batch].mean(dim=1)
        g_k = validation_q[validation_batch].mean(dim=1)
        inner_batch = (mu_m * g_k).sum(dim=1)
        mu_norm = torch.linalg.vector_norm(mu_m, dim=1)
        g_norm = torch.linalg.vector_norm(g_k, dim=1)
        denominator = (mu_norm * g_norm).clamp_min(torch.finfo(torch.float64).tiny)
        values.append(inner_batch)
        cosine_values.append(inner_batch / denominator)
    inner = torch.cat(values)
    cosine = torch.cat(cosine_values)
    flips = inner <= 0
    flip_count = int(flips.sum().item())
    result: dict[str, Any] = {
        "M": m,
        "K": k,
        "MK": m * k,
        "R": inner.numel(),
        "flip_count": flip_count,
        "flip_probability": flip_count / inner.numel(),
        "positive_count": int((inner > 0).sum().item()),
        "inner_product_mean": inner.mean().item(),
        "inner_product_std": inner.std(unbiased=True).item() if inner.numel() > 1 else 0.0,
        "inner_product_min": inner.min().item(),
        "inner_product_max": inner.max().item(),
        "inner_product_median": inner.median().item(),
        "cosine_mean": cosine.mean().item(),
        "cosine_std": cosine.std(unbiased=True).item() if cosine.numel() > 1 else 0.0,
        "cosine_min": cosine.min().item(),
        "cosine_median": cosine.median().item(),
        "cosine_q05": torch.quantile(cosine, 0.05).item(),
    }
    if save_inner_products:
        result["inner_products"] = inner.tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="/root/train-00000-of-00001.parquet")
    parser.add_argument("--validation-data", default="/root/validation-00000-of-00001.parquet")
    parser.add_argument("--model", default="/root/model-merged")
    parser.add_argument("--output", default="/root/flip_probability_results.json")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--records-path", default=None)
    parser.add_argument("--samples", "-B", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=0, help="0 uses all full validation chunks")
    parser.add_argument("--sequence-length", "-L", type=int, default=1024)
    parser.add_argument("--pairs", default="16,32,64,128", help="Comma-separated M:K pairs; 64 means 64:64")
    parser.add_argument("--repetitions", "-R", type=int, default=1000)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--trial-batch-size", type=int, default=50)
    parser.add_argument("--save-inner-products", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.sequence_length < 2 or args.repetitions < 1:
        raise ValueError("B and sequence length must be >=2; R must be >=1")
    if args.trial_batch_size < 1:
        raise ValueError("--trial-batch-size must be positive")
    pairs = parse_pairs(args.pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train_records = load_records(Path(args.train_data), args.records_path)
    validation_records = load_records(Path(args.validation_data), args.records_path)
    train_dataset = FixedTokenDataset(
        train_records, tokenizer, args.text_field, args.sequence_length, args.samples, concatenate=True
    )
    validation_dataset = FixedTokenDataset(
        validation_records,
        tokenizer,
        args.text_field,
        args.sequence_length,
        args.validation_samples,
        concatenate=True,
    )
    validation_size = len(validation_dataset)
    splits = build_splits(args.samples, validation_size, pairs, args.repetitions, args.seed)

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)

    layers = find_layers(model)
    layer_indices = parse_layer_indices(args.layers, len(layers))
    collector = GradientCollector({index: layers[index] for index in layer_indices})
    embedding_handle = model.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.requires_grad_(True)
    )
    def collect_q(dataset: FixedTokenDataset, label: str) -> tuple[dict[int, torch.Tensor], list[float]]:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        rows: dict[int, list[torch.Tensor]] = {index: [] for index in layer_indices}
        losses: list[float] = []
        for sample_index, (input_ids, _stream_index) in enumerate(loader):
            collector.clear()
            input_ids = input_ids.to(args.device, non_blocking=True)
            attention_mask = torch.ones_like(input_ids)
            model.zero_grad(set_to_none=True)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite {label} loss at sample {sample_index}")
            output.loss.backward()
            for layer_index in layer_indices:
                gradient = collector.gradients.get(layer_index)
                if gradient is None:
                    raise RuntimeError(f"No gradient captured for layer {layer_index}")
                # Explicit clone guarantees each Q row owns independent storage.
                # HF causal-LM loss is already averaged over valid tokens. Sum
                # token contributions here instead of introducing another 1/T.
                q = gradient[0].float().sum(dim=0).detach().cpu().clone()
                rows[layer_index].append(q)
            losses.append(output.loss.detach().float().item())
            print(
                f"set={label} sample={sample_index + 1}/{len(dataset)} loss={losses[-1]:.6f}",
                flush=True,
            )
        return {index: torch.stack(values) for index, values in rows.items()}, losses

    train_q, train_losses = collect_q(train_dataset, "train")
    validation_q, validation_losses = collect_q(validation_dataset, "validation")

    collector.close()
    embedding_handle.remove()

    model_name = Path(args.model).name
    hidden_width = int(model.config.hidden_size)
    results: list[dict[str, Any]] = []
    gradient_reuse_checks: list[dict[str, Any]] = []
    for layer_index in layer_indices:
        q = train_q[layer_index]
        validation_layer_q = validation_q[layer_index]
        train_q_norms = torch.linalg.vector_norm(q.double(), dim=1)
        validation_q_norms = torch.linalg.vector_norm(validation_layer_q.double(), dim=1)
        check_indices = sorted(set((1, min(100, q.shape[0] - 1))))
        gradient_reuse_checks.append(
            {
                "layer_id": layer_index,
                "train_q_0_minus_other_l2": {
                    str(index): torch.linalg.vector_norm(q[0] - q[index]).item()
                    for index in check_indices
                },
                "validation_q_0_minus_q_1_l2": torch.linalg.vector_norm(
                    validation_layer_q[0] - validation_layer_q[1]
                ).item(),
                "train_exact_duplicate_rows": int(q.shape[0] - torch.unique(q, dim=0).shape[0]),
                "validation_exact_duplicate_rows": int(
                    validation_layer_q.shape[0] - torch.unique(validation_layer_q, dim=0).shape[0]
                ),
                "train_q_norm_mean": train_q_norms.mean().item(),
                "train_q_norm_min": train_q_norms.min().item(),
                "train_q_norm_max": train_q_norms.max().item(),
                "validation_q_norm_mean": validation_q_norms.mean().item(),
                "validation_q_norm_min": validation_q_norms.min().item(),
                "validation_q_norm_max": validation_q_norms.max().item(),
            }
        )
        for m, k in pairs:
            stats = summarize_inner_products(
                q,
                splits[(m, k)],
                validation_layer_q,
                m,
                k,
                args.trial_batch_size,
                args.save_inner_products,
            )
            record = {
                "model_name": model_name,
                "hidden_width": hidden_width,
                "layer_id": layer_index,
                "train_num_sequences": args.samples,
                "validation_num_sequences": validation_size,
                "sequence_length": args.sequence_length,
                **stats,
            }
            results.append(record)
            print(
                f"layer={layer_index} M={m} K={k} flips={stats['flip_count']}/{args.repetitions} "
                f"p_flip={stats['flip_probability']:.6f}",
                flush=True,
            )

    report = {
        "definition": {
            "mu_M": "mean of M activation-gradient vectors",
            "g_K": "mean of K activation-gradient vectors from the independent validation corpus",
            "is_flip": "dot(mu_M, g_K) <= 0",
            "flip_probability": "flip_count / R",
            "q_i": "sum over token dimension of gradients from HF mean-token loss",
        },
        "model": args.model,
        "train_data": args.train_data,
        "validation_data": args.validation_data,
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "B": args.samples,
        "validation_pool_size": validation_size,
        "sequence_length": args.sequence_length,
        "R": args.repetitions,
        "seed": args.seed,
        "pairs": [{"M": m, "K": k, "MK": m * k} for m, k in pairs],
        "train_average_loss": sum(train_losses) / len(train_losses),
        "validation_average_loss": sum(validation_losses) / len(validation_losses),
        "train_discarded_tokens_after_B_sequences": train_dataset.discarded_tail_tokens,
        "validation_discarded_tail_tokens": validation_dataset.discarded_tail_tokens,
        "gradient_reuse_checks": gradient_reuse_checks,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
