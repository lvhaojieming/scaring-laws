#!/usr/bin/env python3
"""Load a local Hugging Face model on GPU and run a short generation test."""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/model", help="Local model directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--prompt",
        default="artificial intelligence is",
        help="Text used for generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; detected {torch.cuda.device_count()} GPU(s)")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {args.model}")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    torch.cuda.synchronize(device)

    print(f"Load time: {time.perf_counter() - started:.2f} s")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print(f"GPU allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Use total max_length instead of max_new_tokens because some model configs
    # already define max_length, which makes Transformers emit a harmless but
    # confusing conflict warning.
    max_length = inputs.input_ids.shape[1] + args.max_new_tokens
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device)

    generated_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time: {elapsed:.2f} s")
    print(f"Speed: {generated_tokens / elapsed:.2f} tokens/s")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
    print(f"Output: {text}")


if __name__ == "__main__":
    main()
