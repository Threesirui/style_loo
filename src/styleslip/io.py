"""Small, schema-tolerant readers for standalone StyleSlip extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True, slots=True)
class TextRecord:
    text: str
    label: int | None = None
    id: str = ""
    model: str = ""
    source: str = ""
    group_id: str = ""
    attack: str = ""


def text_fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _binary_label(value: Any, *, location: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().casefold()
    labels = {
        "0": 0,
        "human": 0,
        "human-written": 0,
        "human_written": 0,
        "1": 1,
        "ai": 1,
        "machine": 1,
        "generated": 1,
        "chatgpt": 1,
    }
    if normalized not in labels:
        raise ValueError(f"{location}: label must be binary 0/1 or human/AI")
    return labels[normalized]


def _record(raw: Mapping[str, Any], *, text_column: str, label_column: str, location: str) -> TextRecord:
    text = raw.get(text_column)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{location}: {text_column!r} must be a non-empty string")
    return TextRecord(
        text=text,
        label=_binary_label(raw.get(label_column), location=location),
        id=str(raw.get("id", "")),
        model=str(raw.get("model", "")),
        source=str(raw.get("source", raw.get("domain", ""))),
    )


def iter_records(
    path: Path,
    *,
    input_format: str = "auto",
    text_column: str = "text",
    label_column: str = "label",
) -> Iterator[TextRecord]:
    """Read JSONL or CSV records while keeping only common metadata fields."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    resolved = input_format
    if resolved == "auto":
        resolved = "csv" if path.suffix.casefold() == ".csv" else "jsonl"
    if resolved == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                yield _record(
                    row,
                    text_column=text_column,
                    label_column=label_column,
                    location=f"{path} row {row_number}",
                )
        return
    if resolved != "jsonl":
        raise ValueError("input_format must be auto, jsonl, or csv")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number}: invalid JSON") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path} line {line_number}: expected a JSON object")
            yield _record(
                raw,
                text_column=text_column,
                label_column=label_column,
                location=f"{path} line {line_number}",
            )


__all__ = ["TextRecord", "iter_records", "text_fingerprint"]
