#!/usr/bin/env python3
"""Measure transformer-layer activations and activation gradients on JSON/Parquet data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class JsonTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        text_field: str | None,
        target_field: str | None,
        records_path: str | None,
    ) -> None:
        records = load_records(Path(path), records_path)
        self.text_field = text_field
        self.target_field = target_field
        self.records: list[Any] = []
        self.skipped_empty = 0
        for item in records:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = pick_text(item, text_field, ("text", "prompt", "content", "instruction", "question"))
            else:
                raise TypeError(f"Records must be strings or objects, got {type(item).__name__}")
            if not text.strip():
                self.skipped_empty += 1
            else:
                self.records.append(item)
        if not self.records:
            raise ValueError(f"No records found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        item = self.records[index]
        if isinstance(item, str):
            return {"text": item, "target": ""}
        if not isinstance(item, dict):
            raise TypeError(f"Record {index} must be a string or object, got {type(item).__name__}")

        text = pick_text(item, self.text_field, ("text", "prompt", "content", "instruction", "question"))
        target = pick_text(
            item, self.target_field, ("target", "response", "answer", "output", "completion"), required=False
        )
        return {"text": text, "target": target}


def nested_get(value: Any, dotted_path: str) -> Any:
    for key in dotted_path.split("."):
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and key.isdigit():
            value = value[int(key)]
        else:
            raise KeyError(f"Cannot resolve records path at {key!r}")
    return value


def load_records(path: Path, records_path: str | None) -> list[Any]:
    if path.suffix.lower() in (".parquet", ".pq"):
        if records_path:
            raise ValueError("--records-path is for nested JSON and cannot be used with Parquet")
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow: pip install pyarrow") from error
        records = pq.read_table(path).to_pylist()
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if records_path:
            value = nested_get(value, records_path)
        if isinstance(value, list):
            records = value
        elif isinstance(value, dict):
            # A single JSON object is treated as one sample.
            records = [value]
        else:
            raise TypeError("JSON root (or --records-path value) must be an array or object")
    return records


def pick_text(
    item: dict[str, Any], explicit: str | None, candidates: Iterable[str], required: bool = True
) -> str:
    keys = (explicit,) if explicit else candidates
    for key in keys:
        if key and key in item and item[key] is not None:
            value = item[key]
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)
    if required:
        raise KeyError(f"No text field found. Available fields: {list(item)}; use --text-field")
    return ""


class CausalLMCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token or ""
        texts = [s["text"] + s["target"] + eos for s in samples]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # If a target exists, compute loss only on target tokens. Otherwise use
        # ordinary next-token loss over the complete text.
        for row, sample in enumerate(samples):
            if sample["target"]:
                prompt_ids = self.tokenizer(
                    sample["text"], truncation=True, max_length=self.max_length, add_special_tokens=True
                )["input_ids"]
                labels[row, : min(len(prompt_ids), labels.shape[1])] = -100
        batch["labels"] = labels
        return batch


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    floating = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "numel": detached.numel(),
        "bytes": detached.numel() * detached.element_size(),
        "mean_abs": floating.abs().mean().item(),
        "max_abs": floating.abs().max().item(),
        "l2_norm": torch.linalg.vector_norm(floating).item(),
        "rms": floating.square().mean().sqrt().item(),
    }


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for part in value:
            found = first_tensor(part)
            if found is not None:
                return found
    return None


class ActivationRecorder:
    def __init__(self) -> None:
        self.current: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def attach(self, named_modules: list[tuple[str, torch.nn.Module]]) -> None:
        for name, module in named_modules:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                return
            self.current[name] = {"activation": tensor_stats(tensor)}
            if tensor.requires_grad:
                tensor.register_hook(lambda grad, layer=name: self._save_gradient(layer, grad))
        return hook

    def _save_gradient(self, name: str, gradient: torch.Tensor) -> None:
        self.current.setdefault(name, {})["gradient"] = tensor_stats(gradient)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def transformer_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    candidates = (
        "model.layers",                 # Llama/Mistral/Qwen
        "transformer.h",                # GPT-2/Bloom
        "gpt_neox.layers",              # GPT-NeoX
        "model.decoder.layers",         # OPT
    )
    for path in candidates:
        value: Any = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return [(f"layer_{i:02d}", layer) for i, layer in enumerate(value)]
    raise ValueError("Could not find transformer layers for this architecture")


def aggregate(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latest: dict[str, dict[str, Any]] = {}
    for batch in batch_results:
        for layer, kinds in batch["layers"].items():
            latest.setdefault(layer, {})
            for kind, stats in kinds.items():
                latest[layer][kind] = {k: stats[k] for k in ("shape", "dtype", "numel", "bytes")}
                for metric in ("mean_abs", "max_abs", "l2_norm", "rms"):
                    values[f"{layer}.{kind}"][metric].append(stats[metric])

    result: dict[str, Any] = {}
    for layer, kinds in latest.items():
        result[layer] = {}
        for kind, fixed in kinds.items():
            prefix = f"{layer}.{kind}"
            result[layer][kind] = fixed | {
                f"average_{metric}": sum(nums) / len(nums)
                for metric, nums in values[prefix].items()
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSON, JSONL, or Parquet dataset path")
    parser.add_argument("--model", default="/root/model-merged")
    parser.add_argument("--output", default="/root/activation_gradient_results.json")
    parser.add_argument("--text-field", help="Input text field; auto-detected when omitted")
    parser.add_argument("--target-field", help="Optional supervised target field")
    parser.add_argument("--records-path", help="Dotted path to a nested JSON array, e.g. data.train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 2:
        raise ValueError("--batch-size must be >= 1 and --max-length must be >= 2")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.train()
    model.config.use_cache = False
    model.requires_grad_(False)

    # Frozen parameters save memory. This hook makes the embedding output the
    # differentiable root, so activation gradients are still computed.
    embedding_handle = model.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.requires_grad_(True)
    )
    recorder = ActivationRecorder()
    recorder.attach(transformer_layers(model))

    dataset = JsonTextDataset(args.data, args.text_field, args.target_field, args.records_path)
    print(f"dataset_samples={len(dataset)} skipped_empty={dataset.skipped_empty}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CausalLMCollator(tokenizer, args.max_length),
        pin_memory=args.device.startswith("cuda"),
    )

    results: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        recorder.current = {}
        batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        if not torch.isfinite(outputs.loss):
            raise RuntimeError(f"Non-finite loss in batch {batch_index}: {outputs.loss.item()}")
        outputs.loss.backward()
        results.append(
            {
                "batch": batch_index,
                "batch_size": batch["input_ids"].shape[0],
                "sequence_length": batch["input_ids"].shape[1],
                "loss": outputs.loss.detach().float().item(),
                "layers": recorder.current,
            }
        )
        print(f"batch={batch_index} loss={results[-1]['loss']:.6f}")

    recorder.close()
    embedding_handle.remove()
    if not results:
        raise RuntimeError("No batches were processed")

    report = {
        "model": args.model,
        "data": args.data,
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "samples": sum(item["batch_size"] for item in results),
        "dataset_samples": len(dataset),
        "skipped_empty_samples": dataset.skipped_empty,
        "batches": len(results),
        "average_loss": sum(item["loss"] for item in results) / len(results),
        "summary": aggregate(results),
        "per_batch": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
