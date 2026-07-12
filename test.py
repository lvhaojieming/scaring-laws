#!/usr/bin/env python3
"""
Evaluate a local Hugging Face causal language model across multiple dimensions
and write all generations, automatic scores, speed, and GPU memory usage to JSON.

Designed primarily for a base pretrained model rather than an instruction-tuned model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TESTS: list[dict[str, Any]] = [
    # 1. English continuation and style
    {
        "id": "continuation_news",
        "dimension": "continuation",
        "prompt": (
            "The government announced a new energy policy on Tuesday. "
            "According to the official statement, the policy is designed to"
        ),
        "max_new_tokens": 120,
        "scoring": {"type": "manual"},
        "notes": "Check fluency, news style, topic consistency, and factual hallucination.",
    },
    {
        "id": "continuation_story",
        "dimension": "continuation",
        "prompt": (
            "Emily opened the old wooden box and found a letter inside. "
            "The handwriting looked familiar, but"
        ),
        "max_new_tokens": 120,
        "scoring": {"type": "manual"},
        "notes": "Check narrative continuity, tense consistency, and character consistency.",
    },
    {
        "id": "continuation_academic",
        "dimension": "continuation",
        "prompt": (
            "Recent advances in machine learning have significantly improved "
            "the performance of language models. However, increasing model size "
            "alone does not necessarily"
        ),
        "max_new_tokens": 120,
        "scoring": {"type": "manual"},
        "notes": "Check academic style, technical coherence, and topic preservation.",
    },

    # 2. Grammar
    {
        "id": "grammar_subject_verb",
        "dimension": "grammar",
        "prompt": "Complete the sentence:\nEach of the students",
        "max_new_tokens": 12,
        "scoring": {
            "type": "contains_any",
            "answers": ["has", "was"],
        },
        "notes": "The continuation should use a singular verb.",
    },
    {
        "id": "grammar_conditional",
        "dimension": "grammar",
        "prompt": "Complete the sentence:\nIf I had known about the meeting earlier, I",
        "max_new_tokens": 20,
        "scoring": {
            "type": "contains_any",
            "answers": ["would have", "could have", "might have"],
        },
        "notes": "Tests the third conditional.",
    },

    # 3. Commonsense
    {
        "id": "commonsense_freezing",
        "dimension": "commonsense",
        "prompt": "Question: At standard atmospheric pressure, water freezes at what temperature?\nAnswer:",
        "max_new_tokens": 20,
        "scoring": {
            "type": "contains_any",
            "answers": ["0 degrees celsius", "0°c", "zero degrees celsius", "32 degrees fahrenheit", "32°f"],
        },
    },
    {
        "id": "commonsense_locked_house",
        "dimension": "commonsense",
        "prompt": (
            "Question: A person has forgotten the key to a locked house. "
            "What is a safe and sensible action?\nAnswer:"
        ),
        "max_new_tokens": 40,
        "scoring": {
            "type": "contains_any",
            "answers": ["locksmith", "spare key", "contact the owner", "call the owner"],
        },
    },

    # 4. Logical reasoning
    {
        "id": "reasoning_transitive",
        "dimension": "reasoning",
        "prompt": (
            "Alice is older than Bob. Bob is older than Charlie.\n"
            "Question: Who is older, Alice or Charlie?\nAnswer:"
        ),
        "max_new_tokens": 20,
        "scoring": {"type": "contains_any", "answers": ["alice"]},
    },
    {
        "id": "reasoning_spatial",
        "dimension": "reasoning",
        "prompt": (
            "The red box is to the left of the blue box. "
            "The green box is to the right of the blue box.\n"
            "Question: Is the red box to the left or right of the green box?\nAnswer:"
        ),
        "max_new_tokens": 24,
        "scoring": {"type": "contains_any", "answers": ["left"]},
    },
    {
        "id": "reasoning_negation",
        "dimension": "reasoning",
        "prompt": (
            "All roses are flowers. Some flowers fade quickly. "
            "This information does not imply that all roses fade quickly.\n"
            "Question: Can we conclude that every rose fades quickly?\nAnswer:"
        ),
        "max_new_tokens": 24,
        "scoring": {
            "type": "contains_any",
            "answers": ["no", "cannot", "not necessarily"],
        },
    },

    # 5. Arithmetic
    {
        "id": "math_addition",
        "dimension": "mathematics",
        "prompt": "Question: What is 27 plus 35?\nAnswer:",
        "max_new_tokens": 16,
        "scoring": {"type": "number", "answer": 62},
    },
    {
        "id": "math_multiplication",
        "dimension": "mathematics",
        "prompt": (
            "Question: A box contains 8 rows of 6 apples. "
            "How many apples are there in total?\nAnswer:"
        ),
        "max_new_tokens": 20,
        "scoring": {"type": "number", "answer": 48},
    },
    {
        "id": "math_two_step",
        "dimension": "mathematics",
        "prompt": (
            "Question: A train travels 60 miles per hour for 3 hours. "
            "How far does it travel?\nAnswer:"
        ),
        "max_new_tokens": 24,
        "scoring": {"type": "number", "answer": 180},
    },

    # 6. Factual knowledge
    {
        "id": "knowledge_capital",
        "dimension": "knowledge",
        "prompt": "Question: What is the capital of France?\nAnswer:",
        "max_new_tokens": 16,
        "scoring": {"type": "contains_any", "answers": ["paris"]},
    },
    {
        "id": "knowledge_shakespeare",
        "dimension": "knowledge",
        "prompt": "Question: Who wrote Romeo and Juliet?\nAnswer:",
        "max_new_tokens": 20,
        "scoring": {
            "type": "contains_any",
            "answers": ["william shakespeare", "shakespeare"],
        },
    },
    {
        "id": "knowledge_photosynthesis",
        "dimension": "knowledge",
        "prompt": "Question: What is photosynthesis?\nAnswer:",
        "max_new_tokens": 64,
        "scoring": {
            "type": "keyword_fraction",
            "answers": ["plants", "light", "carbon dioxide", "water", "glucose", "oxygen"],
            "minimum_fraction": 0.5,
        },
    },

    # 7. Code completion
    {
        "id": "code_add",
        "dimension": "code",
        "prompt": "Complete the Python function:\n\ndef add_numbers(a, b):\n    return",
        "max_new_tokens": 20,
        "scoring": {
            "type": "regex",
            "pattern": r"\ba\s*\+\s*b\b",
        },
    },
    {
        "id": "code_even",
        "dimension": "code",
        "prompt": "Complete the Python function:\n\ndef is_even(n):\n    return",
        "max_new_tokens": 24,
        "scoring": {
            "type": "regex",
            "pattern": r"\bn\s*%\s*2\s*==\s*0\b",
        },
    },

    # 8. Long-context retrieval
    {
        "id": "context_retrieval_bicycle",
        "dimension": "context_retrieval",
        "prompt": (
            "Daniel lives in Boston and works as a software engineer. "
            "His sister, Laura, lives in Seattle and works as a doctor. "
            "Daniel owns a black bicycle, while Laura owns a red car. "
            "Several months later, the family decided to meet during the summer. "
            "Daniel planned to travel across the country to visit his sister.\n"
            "Question: What color is Daniel's bicycle?\nAnswer:"
        ),
        "max_new_tokens": 20,
        "scoring": {"type": "contains_any", "answers": ["black"]},
    },
    {
        "id": "context_retrieval_city",
        "dimension": "context_retrieval",
        "prompt": (
            "Daniel lives in Boston and works as a software engineer. "
            "His sister, Laura, lives in Seattle and works as a doctor. "
            "Daniel owns a black bicycle, while Laura owns a red car. "
            "Several months later, the family decided to meet during the summer. "
            "Daniel planned to travel across the country to visit his sister.\n"
            "Question: Where does Laura live?\nAnswer:"
        ),
        "max_new_tokens": 20,
        "scoring": {"type": "contains_any", "answers": ["seattle"]},
    },

    # 9. Temporal consistency
    {
        "id": "temporal_consistency",
        "dimension": "consistency",
        "prompt": (
            "The following article is set in 2026. The policy sets targets only "
            "for 2030 and 2035, and it does not mention any earlier deadline.\n\n"
            "The government announced a new energy policy on Tuesday. "
            "According to the official statement, the policy is designed to"
        ),
        "max_new_tokens": 120,
        "scoring": {
            "type": "forbidden_any",
            "answers": ["2015", "2020", "2021", "2022", "2023", "2024", "2025"],
        },
        "notes": "Tests whether the continuation respects explicit temporal constraints.",
    },

    # 10. Repetition and degeneration
    {
        "id": "repetition_long_generation",
        "dimension": "repetition",
        "prompt": "Machine learning is a field of artificial intelligence that",
        "max_new_tokens": 200,
        "scoring": {"type": "repetition_metrics"},
        "notes": "No repetition penalty should be used in diagnostic mode.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/model", help="Local model directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--output",
        default="model_capability_results.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["greedy", "sample"],
        default="greedy",
        help=(
            "Use greedy for fair deterministic comparison. "
            "Use sample for qualitative generation."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Keep at 1.0 for diagnostic testing so repetition is not hidden.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Keep at 0 for diagnostic testing.",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=None,
        help="Override every test's max_new_tokens when set.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from the local model directory.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    return float(match.group(0)) if match else None


def ngram_repetition_ratio(token_ids: list[int], n: int) -> float:
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def maximum_ngram_count(token_ids: list[int], n: int) -> int:
    if len(token_ids) < n:
        return 0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return max(Counter(ngrams).values(), default=0)


def score_output(
    generated_text: str,
    generated_token_ids: list[int],
    scoring: dict[str, Any],
) -> dict[str, Any]:
    kind = scoring["type"]
    normalized = normalize_text(generated_text)

    if kind == "manual":
        return {"score": None, "passed": None, "method": "manual_review"}

    if kind == "contains_any":
        answers = [normalize_text(x) for x in scoring["answers"]]
        matched = [answer for answer in answers if answer in normalized]
        return {
            "score": 1.0 if matched else 0.0,
            "passed": bool(matched),
            "matched": matched,
            "expected_any": scoring["answers"],
        }

    if kind == "forbidden_any":
        forbidden = [normalize_text(x) for x in scoring["answers"]]
        matched = [item for item in forbidden if item in normalized]
        return {
            "score": 0.0 if matched else 1.0,
            "passed": not matched,
            "forbidden_matched": matched,
        }

    if kind == "number":
        predicted = extract_first_number(generated_text)
        expected = float(scoring["answer"])
        passed = predicted is not None and math.isclose(predicted, expected, rel_tol=0.0, abs_tol=1e-9)
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "predicted_number": predicted,
            "expected_number": expected,
        }

    if kind == "regex":
        match = re.search(scoring["pattern"], generated_text, flags=re.IGNORECASE | re.MULTILINE)
        return {
            "score": 1.0 if match else 0.0,
            "passed": bool(match),
            "matched": match.group(0) if match else None,
            "pattern": scoring["pattern"],
        }

    if kind == "keyword_fraction":
        answers = [normalize_text(x) for x in scoring["answers"]]
        matched = [answer for answer in answers if answer in normalized]
        fraction = len(matched) / len(answers)
        minimum = float(scoring.get("minimum_fraction", 1.0))
        return {
            "score": fraction,
            "passed": fraction >= minimum,
            "matched": matched,
            "expected_keywords": scoring["answers"],
            "minimum_fraction": minimum,
        }

    if kind == "repetition_metrics":
        metrics = {
            f"repeat_{n}gram_ratio": round(ngram_repetition_ratio(generated_token_ids, n), 6)
            for n in (2, 3, 4, 8)
        }
        metrics.update(
            {
                f"max_{n}gram_count": maximum_ngram_count(generated_token_ids, n)
                for n in (2, 3, 4, 8)
            }
        )
        # This is a diagnostic heuristic, not a universal quality threshold.
        passed = (
            metrics["repeat_4gram_ratio"] < 0.20
            and metrics["max_8gram_count"] <= 2
        )
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            **metrics,
        }

    raise ValueError(f"Unknown scoring type: {kind}")


def build_dimension_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(item["dimension"], []).append(item)

    summary: dict[str, Any] = {}
    for dimension, items in grouped.items():
        scored = [
            item["evaluation"]["score"]
            for item in items
            if item["evaluation"]["score"] is not None
        ]
        passed = [
            item["evaluation"]["passed"]
            for item in items
            if item["evaluation"]["passed"] is not None
        ]
        summary[dimension] = {
            "tests": len(items),
            "automatically_scored_tests": len(scored),
            "mean_score": round(sum(scored) / len(scored), 6) if scored else None,
            "passed": sum(bool(x) for x in passed),
            "failed": sum(not bool(x) for x in passed),
        }
    return summary


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU {args.gpu}; detected {torch.cuda.device_count()} GPU(s)"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model).resolve()),
        "gpu_index": args.gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "dtype": str(next(model.parameters()).dtype),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_time_seconds": round(load_seconds, 6),
        "gpu_memory_after_load_gib": round(
            torch.cuda.memory_allocated(device) / 1024**3, 6
        ),
        "transformers_generation": {
            "mode": args.mode,
            "seed": args.seed,
            "temperature": args.temperature if args.mode == "sample" else None,
            "top_p": args.top_p if args.mode == "sample" else None,
            "top_k": args.top_k if args.mode == "sample" else None,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
    }

    results: list[dict[str, Any]] = []
    total_generated_tokens = 0
    total_generation_seconds = 0.0

    for index, test in enumerate(TESTS, start=1):
        prompt = test["prompt"]
        max_new_tokens = (
            args.default_max_new_tokens
            if args.default_max_new_tokens is not None
            else test["max_new_tokens"]
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": args.mode == "sample",
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "pad_token_id": pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.mode == "sample":
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        generated_ids = output_ids[0, input_length:]
        generated_token_ids = generated_ids.tolist()
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        full_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generated_tokens = len(generated_token_ids)
        tokens_per_second = generated_tokens / elapsed if elapsed > 0 else None
        ended_with_eos = (
            tokenizer.eos_token_id is not None
            and bool(generated_token_ids)
            and generated_token_ids[-1] == tokenizer.eos_token_id
        )

        evaluation = score_output(
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            scoring=test["scoring"],
        )

        result = {
            "id": test["id"],
            "dimension": test["dimension"],
            "prompt": prompt,
            "continuation": generated_text,
            "full_output": full_text,
            "input_tokens": input_length,
            "generated_tokens": generated_tokens,
            "max_new_tokens": max_new_tokens,
            "ended_with_eos": ended_with_eos,
            "generation_time_seconds": round(elapsed, 6),
            "tokens_per_second": round(tokens_per_second, 6)
            if tokens_per_second is not None
            else None,
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated(device) / 1024**3, 6
            ),
            "evaluation": evaluation,
            "notes": test.get("notes"),
        }
        results.append(result)

        total_generated_tokens += generated_tokens
        total_generation_seconds += elapsed

        status = evaluation["passed"]
        if status is True:
            status_text = "PASS"
        elif status is False:
            status_text = "FAIL"
        else:
            status_text = "MANUAL"

        print(
            f"[{index:02d}/{len(TESTS):02d}] "
            f"{test['dimension']}/{test['id']} | {status_text} | "
            f"{generated_tokens} tokens | {tokens_per_second:.2f} tok/s"
        )
        print(f"  Output: {generated_text[:240]!r}")

    dimension_summary = build_dimension_summary(results)
    all_auto_scores = [
        item["evaluation"]["score"]
        for item in results
        if item["evaluation"]["score"] is not None
    ]

    report = {
        "metadata": metadata,
        "summary": {
            "total_tests": len(results),
            "automatically_scored_tests": len(all_auto_scores),
            "overall_mean_automatic_score": round(
                sum(all_auto_scores) / len(all_auto_scores), 6
            )
            if all_auto_scores
            else None,
            "total_generated_tokens": total_generated_tokens,
            "total_generation_time_seconds": round(
                total_generation_seconds, 6
            ),
            "aggregate_tokens_per_second": round(
                total_generated_tokens / total_generation_seconds, 6
            )
            if total_generation_seconds > 0
            else None,
            "dimensions": dimension_summary,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print(f"Saved JSON report to: {output_path.resolve()}")
    print(
        "Overall automatic score: "
        f"{report['summary']['overall_mean_automatic_score']}"
    )


if __name__ == "__main__":
    main()
