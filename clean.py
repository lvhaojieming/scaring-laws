from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


DEFAULT_INPUT_DIR = Path("/datadisk_1/balanced_web_edu_mix_10B/jsonl")
DEFAULT_OUTPUT_DIR = Path("/datadisk_1/balanced_web_edu_mix_10B/cleaned_v1")

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_RE = re.compile(r"(?x)(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]")
ALPHA_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{19,}")


@dataclass(frozen=True)
class CleaningConfig:
    min_chars: int = 200
    max_chars: int = 200_000
    min_words: int = 30
    max_url_ratio: float = 0.20
    min_alpha_ratio: float = 0.25
    max_non_printable_ratio: float = 0.02
    max_repeated_line_ratio: float = 0.30
    max_repeated_paragraph_ratio: float = 0.40
    max_short_line_ratio: float = 0.70
    min_avg_word_len: float = 2.0
    max_avg_word_len: float = 18.0
    redact_pii: bool = True
    progress_every: int = 50_000


class NormalizeAndRedactFormatter(BaseFormatter):
    name = "normalize_redact"

    def __init__(self, redact_pii: bool = True):
        super().__init__()
        self.redact_pii = redact_pii

    def format(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = CONTROL_CHARS_RE.sub(" ", text)
        text = "\n".join(HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
        text = MANY_BLANK_LINES_RE.sub("\n\n", text).strip()

        if self.redact_pii:
            text = EMAIL_RE.sub("[EMAIL]", text)
            text = IPV4_RE.sub("[IP]", text)
            text = PHONE_RE.sub("[PHONE]", text)
        return text


class LightweightQualityFilter(BaseFilter):
    name = "lightweight_quality"

    def __init__(self, config: CleaningConfig, exclusion_writer: JsonlWriter | None = None):
        super().__init__(exclusion_writer=exclusion_writer)
        self.config = config
        self._processed = 0
        self._kept = 0
        self._dropped = 0
        self._started_at = time.time()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        self._processed += 1
        result = self._filter(doc)
        keep = result if isinstance(result, bool) else result[0]
        if keep:
            self._kept += 1
        else:
            self._dropped += 1
        self._maybe_log_progress()
        return result

    def _filter(self, doc: Document) -> bool | tuple[bool, str]:
        text = doc.text
        char_count = len(text)
        if char_count < self.config.min_chars:
            return False, "too_short"
        if char_count > self.config.max_chars:
            return False, "too_long"

        non_printable = sum(1 for ch in text if not ch.isprintable() and ch != "\n")
        if non_printable / max(char_count, 1) > self.config.max_non_printable_ratio:
            return False, "non_printable_ratio"

        words = WORD_RE.findall(text)
        word_count = len(words)
        if word_count < self.config.min_words:
            return False, "too_few_words"

        avg_word_len = sum(len(word) for word in words) / max(word_count, 1)
        if avg_word_len < self.config.min_avg_word_len:
            return False, "avg_word_too_short"
        if avg_word_len > self.config.max_avg_word_len:
            return False, "avg_word_too_long"

        alpha_ratio = len(ALPHA_RE.findall(text)) / max(char_count, 1)
        if alpha_ratio < self.config.min_alpha_ratio:
            return False, "low_alpha_ratio"

        url_count = len(URL_RE.findall(text))
        if url_count / max(word_count, 1) > self.config.max_url_ratio:
            return False, "url_ratio"

        if REPEATED_CHAR_RE.search(text):
            return False, "repeated_character_run"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            short_line_ratio = sum(1 for line in lines if len(line) < 30) / len(lines)
            if len(lines) >= 8 and short_line_ratio > self.config.max_short_line_ratio:
                return False, "short_line_ratio"

            repeated_line_ratio = repeated_item_ratio(lines)
            if repeated_line_ratio > self.config.max_repeated_line_ratio:
                return False, "repeated_line_ratio"

        paragraphs = [para.strip() for para in text.split("\n\n") if len(para.strip()) >= 40]
        if len(paragraphs) >= 4 and repeated_item_ratio(paragraphs) > self.config.max_repeated_paragraph_ratio:
            return False, "repeated_paragraph_ratio"

        doc.metadata["cleaning_v1_chars"] = char_count
        doc.metadata["cleaning_v1_words"] = word_count
        doc.metadata["cleaning_v1_alpha_ratio"] = round(alpha_ratio, 4)
        return True

    def _maybe_log_progress(self) -> None:
        if self.config.progress_every <= 0:
            return
        if self._processed % self.config.progress_every != 0:
            return
        elapsed = max(time.time() - self._started_at, 1e-6)
        rate = self._processed / elapsed
        print(
            f"[cleaning_v1] processed={self._processed:,} kept={self._kept:,} "
            f"dropped={self._dropped:,} rate={rate:,.0f} docs/s",
            flush=True,
        )


def repeated_item_ratio(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(items)


def reader_adapter(reader: JsonlReader, data: dict[str, Any], path: str, id_in_file: int | str) -> dict[str, Any]:
    source = str(data.get("source") or Path(path).stem)
    source_repo = str(data.get("source_repo") or "")
    stable_key = f"{path}:{id_in_file}:{source}:{data.get('text', '')[:256]}"
    doc_id = data.get("id") or hashlib.sha1(stable_key.encode("utf-8", errors="ignore")).hexdigest()
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    metadata = {
        **metadata,
        "source": source,
        "source_repo": source_repo,
        "input_file": path,
        "input_line": int(id_in_file) if isinstance(id_in_file, int) else str(id_in_file),
        "cleaning_version": "v1",
    }
    return {"text": str(data.get("text") or ""), "id": str(doc_id), "metadata": metadata}


def writer_adapter(*args) -> dict[str, Any]:
    document = args[-1]
    return {
        "text": document.text,
        "source": document.metadata.get("source", ""),
        "source_repo": document.metadata.get("source_repo", ""),
        "id": document.id,
        "metadata": document.metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1, help="Debug limit per reader shard; -1 means no limit.")
    parser.add_argument("--glob-pattern", default="*.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-chars", type=int, default=CleaningConfig.min_chars)
    parser.add_argument("--max-chars", type=int, default=CleaningConfig.max_chars)
    parser.add_argument("--min-words", type=int, default=CleaningConfig.min_words)
    parser.add_argument("--progress-every", type=int, default=CleaningConfig.progress_every)
    parser.add_argument("--no-redact-pii", action="store_true")
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Also save rejected documents with metadata.filter_reason. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    cleaned_dir = args.output_dir / "jsonl"
    rejected_dir = args.output_dir / "rejected"
    logs_dir = args.output_dir / "logs"
    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = list(cleaned_dir.glob("*.jsonl")) if cleaned_dir.exists() else []
    if existing_outputs and not args.overwrite:
        raise SystemExit(
            f"Output already exists under {cleaned_dir}. Pass --overwrite or choose a different --output-dir."
        )
    if args.overwrite:
        for folder in (cleaned_dir, rejected_dir, logs_dir):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        path.unlink()

    config = CleaningConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_words=args.min_words,
        redact_pii=not args.no_redact_pii,
        progress_every=args.progress_every,
    )

    input_files = sorted(str(path) for path in args.input_dir.glob(args.glob_pattern))
    if not input_files:
        raise SystemExit(f"No input files matched {args.input_dir}/{args.glob_pattern}")

    exclusion_writer = None
    if args.save_rejected:
        exclusion_writer = JsonlWriter(
            str(rejected_dir),
            output_filename="rejected_${rank}.jsonl",
            compression=None,
            adapter=writer_adapter,
        )

    pipeline = [
        JsonlReader(
            str(args.input_dir),
            glob_pattern=args.glob_pattern,
            adapter=reader_adapter,
            limit=args.limit,
            file_progress=True,
        ),
        NormalizeAndRedactFormatter(redact_pii=config.redact_pii),
        LightweightQualityFilter(config=config, exclusion_writer=exclusion_writer),
        JsonlWriter(
            str(cleaned_dir),
            output_filename="cleaned_${rank}.jsonl",
            compression=None,
            adapter=writer_adapter,
        ),
    ]

    manifest = {
        "pipeline": "balanced_web_edu_mix_cleaning_v1",
        "datatrove_version": get_datatrove_version(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "cleaned_dir": str(cleaned_dir),
        "rejected_dir": str(rejected_dir) if args.save_rejected else None,
        "save_rejected": args.save_rejected,
        "logs_dir": str(logs_dir),
        "input_files": input_files,
        "tasks": args.tasks,
        "workers": args.workers,
        "glob_pattern": args.glob_pattern,
        "limit": args.limit,
        "rules": asdict(config),
        "notes": [
            "No MinHash, semantic deduplication, or classifier is used.",
            "Filtering is rule-based and deterministic.",
            "Only cleaned documents are saved by default.",
            "Pass --save-rejected to also save rejected documents with metadata.filter_reason.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=str(logs_dir),
        skip_completed=not args.overwrite,
        start_method="fork",
    )
    executor.run()


def get_datatrove_version() -> str:
    try:
        return importlib.metadata.version("datatrove")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
