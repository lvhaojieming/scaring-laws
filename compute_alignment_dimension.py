#!/usr/bin/env python3
"""Compute effective alignment dimensions from hidden-state activation gradients.

For every selected transformer layer, this program builds Q in R^(B x N), where
each row is the token-mean activation gradient from one fixed-length sequence.
Model parameters are frozen: backward computes activation gradients but does not
allocate parameter-gradient buffers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path, records_path: str | None) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        if records_path:
            raise ValueError("--records-path cannot be used with Parquet")
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow") from error
        return pq.read_table(path).to_pylist()

    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if records_path:
        for key in records_path.split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError("JSON root (or --records-path value) must be an array or object")


def get_text(item: Any, text_field: str | None) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"A record must be a string or object, got {type(item).__name__}")
    fields: Iterable[str] = (text_field,) if text_field else (
        "text", "prompt", "content", "instruction", "question"
    )
    for field in fields:
        if field and field in item and item[field] is not None:
            value = item[field]
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    raise KeyError(f"Cannot find text in fields {list(item)}; pass --text-field")


class FixedTokenDataset(Dataset):
    """Select the first B records that tokenize to at least sequence_length tokens."""

    def __init__(
        self,
        records: list[Any],
        tokenizer: Any,
        text_field: str | None,
        sequence_length: int,
        samples: int,
        concatenate: bool = False,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.source_indices: list[int] = []
        self.discarded_tail_tokens = 0
        if concatenate:
            token_stream: list[int] = []
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("Tokenizer has no EOS token for document separation")
            for source_index, record in enumerate(records):
                text = get_text(record, text_field)
                if not text.strip():
                    continue
                token_stream.extend(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
                token_stream.append(eos_id)
            available = len(token_stream) // sequence_length
            take = available if samples == 0 else min(samples, available)
            for index in range(take):
                start = index * sequence_length
                self.input_ids.append(torch.tensor(token_stream[start : start + sequence_length], dtype=torch.long))
                self.source_indices.append(index)
            self.discarded_tail_tokens = len(token_stream) - take * sequence_length
            if samples > 0 and take < samples:
                raise ValueError(
                    f"Requested B={samples}, but the concatenated corpus only contains "
                    f"{available} full sequences of {sequence_length} tokens"
                )
            if len(self.input_ids) < 2:
                raise ValueError("Concatenated corpus must provide at least two full sequences")
            return

        for source_index, record in enumerate(records):
            text = get_text(record, text_field)
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) < sequence_length:
                continue
            self.input_ids.append(torch.tensor(ids[:sequence_length], dtype=torch.long))
            self.source_indices.append(source_index)
            if samples > 0 and len(self.input_ids) == samples:
                break
        if samples == 0:
            samples = len(self.input_ids)
        if len(self.input_ids) < samples:
            raise ValueError(
                f"Requested B={samples} sequences of {sequence_length} tokens, but only "
                f"{len(self.input_ids)} qualifying records were found"
            )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.input_ids[index], self.source_indices[index]


def find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        value: Any = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return list(value)
    raise ValueError("Cannot locate transformer layers for this model architecture")


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for part in value:
            tensor = first_tensor(part)
            if tensor is not None:
                return tensor
    return None


class GradientCollector:
    def __init__(self, selected: dict[int, torch.nn.Module]) -> None:
        self.gradients: dict[int, torch.Tensor] = {}
        self.handles = [module.register_forward_hook(self._hook(index)) for index, module in selected.items()]

    def _hook(self, layer_index: int):
        def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = first_tensor(output)
            if hidden is None:
                raise RuntimeError(f"Layer {layer_index} did not return a tensor")
            hidden.register_hook(
                lambda gradient, index=layer_index: self.gradients.__setitem__(index, gradient.detach())
            )
        return forward_hook

    def clear(self) -> None:
        self.gradients.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_layer_indices(spec: str, layer_count: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(layer_count))
    indices: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        index = int(piece)
        if index < 0:
            index += layer_count
        if not 0 <= index < layer_count:
            raise ValueError(f"Layer {piece} is outside [0, {layer_count - 1}]")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("No target layers selected")
    return indices


def finite_or_string(value: torch.Tensor) -> float | str:
    number = value.item()
    if math.isfinite(number):
        return number
    return "Infinity" if number > 0 else "NaN"


def calculate_metrics(q: torch.Tensor, epsilon: float) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Calculate the requested statistics in float64 for numerical stability."""
    q = q.to(dtype=torch.float64, device="cpu")
    batch_size, width = q.shape
    if batch_size < 2:
        raise ValueError("B must be at least 2 because V and T use B-1")

    mu = q.mean(dim=0)
    signal = mu.square().sum()                         # S = ||mu||_2^2
    x = q - mu                                        # X in R^(B x N)
    projections = x @ mu                              # mu^T x_i
    directional_noise = projections.square().sum() / (batch_size - 1)
    gram = x @ x.T                                    # G = X X^T
    total_noise = gram.square().sum() / ((batch_size - 1) ** 2)
    signal_squared = signal.square()

    d_parallel = signal_squared / (directional_noise + epsilon)
    d2 = signal_squared / (total_noise + epsilon)
    d_align = torch.minimum(d_parallel, d2)

    metrics = {
        "B": batch_size,
        "N": width,
        "S": signal.item(),
        "S_squared": signal_squared.item(),
        "V": directional_noise.item(),
        "T": total_noise.item(),
        "d_parallel": finite_or_string(d_parallel),
        "d2": finite_or_string(d2),
        "d_align": finite_or_string(d_align),
        "mu_l2_norm": torch.linalg.vector_norm(mu).item(),
        "mean_q_l2_norm": torch.linalg.vector_norm(q, dim=1).mean().item(),
        "centered_row_sum_l2": torch.linalg.vector_norm(x.sum(dim=0)).item(),
        "epsilon": epsilon,
    }
    return metrics, q, mu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Parquet, JSON, or JSONL dataset")
    parser.add_argument("--model", default="/root/model-merged")
    parser.add_argument("--output", default="/root/alignment_dimension_results.json")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--records-path", default=None, help="Dotted path for nested JSON only")
    parser.add_argument("--samples", "-B", type=int, default=64)
    parser.add_argument("--sequence-length", "-L", type=int, default=128)
    parser.add_argument("--layers", default="all", help="Comma-separated zero-based indices, or 'all'")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--save-q", action="store_true", help="Include Q and mu arrays in output JSON")
    parser.add_argument("--epsilon", type=float, default=0, help="Denominator stabilizer")
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="Join non-empty documents with EOS and split into fixed-length sequences; -B 0 uses all full chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 0 or (args.samples == 1):
        raise ValueError("--samples/-B must be 0 (all chunks) or at least 2")
    if args.samples == 0 and not args.concatenate:
        raise ValueError("--samples 0 requires --concatenate")
    if args.sequence_length < 2:
        raise ValueError("--sequence-length/-L must be at least 2")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = load_records(Path(args.data), args.records_path)
    dataset = FixedTokenDataset(
        records, tokenizer, args.text_field, args.sequence_length, args.samples, args.concatenate
    )
    actual_samples = len(dataset)
    # batch_size=1 is intentional: the definition requires one forward/backward per sequence.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.train()
    model.config.use_cache = False
    model.requires_grad_(False)

    layers = find_layers(model)
    layer_indices = parse_layer_indices(args.layers, len(layers))
    selected = {index: layers[index] for index in layer_indices}

    # With frozen parameters, make embeddings the differentiable graph root.
    embedding_handle = model.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.requires_grad_(True)
    )
    collector = GradientCollector(selected)
    rows: dict[int, list[torch.Tensor]] = {index: [] for index in layer_indices}
    losses: list[float] = []

    for sample_index, (input_ids, _source_index) in enumerate(loader):
        collector.clear()
        input_ids = input_ids.to(args.device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite loss for selected sample {sample_index}")
        output.loss.backward()

        for layer_index in layer_indices:
            gradient = collector.gradients.get(layer_index)
            if gradient is None:
                raise RuntimeError(f"No activation gradient captured for layer {layer_index}")
            # [1, L, N] -> [N]. All sequences have exactly L tokens, with no padding.
            rows[layer_index].append(gradient[0].float().mean(dim=0).cpu())
        losses.append(output.loss.detach().float().item())
        print(f"sample={sample_index + 1}/{actual_samples} loss={losses[-1]:.6f}", flush=True)

    collector.close()
    embedding_handle.remove()

    layer_results: list[dict[str, Any]] = []
    hidden_width = int(model.config.hidden_size)
    model_name = Path(args.model).name
    for layer_index in layer_indices:
        metrics, q, mu = calculate_metrics(torch.stack(rows[layer_index]), args.epsilon)
        if args.save_q:
            metrics["Q"] = q.tolist()
            metrics["mu"] = mu.tolist()
        record = {
            "model_name": model_name,
            "hidden_width": hidden_width,
            "layer_id": layer_index,
            "num_sequences": actual_samples,
            "sequence_length": args.sequence_length,
            **metrics,
        }
        layer_results.append(record)
        print(
            f"layer={layer_index} N={metrics['N']} "
            f"d_parallel={metrics['d_parallel']} d2={metrics['d2']} d_align={metrics['d_align']}"
        )

    report = {
        "definition": {
            "q_i": "mean over token dimension of the hidden-state activation gradient",
            "S": "||mu||_2^2",
            "V": "sum_i (mu^T x_i)^2 / (B-1)",
            "T": "||X X^T||_F^2 / (B-1)^2",
            "d_parallel": "S^2 / V",
            "d2": "S^2 / T",
            "d_align": "min(d_parallel, d2)",
        },
        "model": args.model,
        "data": args.data,
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "B": actual_samples,
        "sequence_length": args.sequence_length,
        "epsilon": args.epsilon,
        "concatenated_documents": args.concatenate,
        "discarded_tail_tokens": dataset.discarded_tail_tokens,
        "source_row_indices": dataset.source_indices,
        "average_loss": sum(losses) / len(losses),
        "results": layer_results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
