"""Command-line extraction of standalone Style-LOO archives."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import nltk
import numpy as np
from tqdm.auto import tqdm

from .io import TextRecord, iter_records, text_fingerprint
from .style_loo import (
    MODEL_NAME,
    STYLE_CHANNEL_NAMES,
    StyleLooConfig,
    build_style_loo_waves,
)


LOGGER = logging.getLogger("styleslip")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_nltk_data_path(path: Path | None) -> None:
    if path is not None:
        resolved = str(path.expanduser().resolve())
        if resolved not in nltk.data.path:
            nltk.data.path.insert(0, resolved)


def _discover_nltk_data_path() -> Path | None:
    """Find the sibling TextWave resource cache when both projects coexist."""

    candidates = (
        Path.cwd().parent / "Textwave" / ".nltk_data",
        Path.cwd().parent / "TextWave" / ".nltk_data",
        Path(__file__).resolve().parents[3] / "Textwave" / ".nltk_data",
        Path(__file__).resolve().parents[3] / "TextWave" / ".nltk_data",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _load_encoder(args: argparse.Namespace):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "StyleDistance requires sentence-transformers; install with "
            "`python -m pip install -e \".[model]\"`"
        ) from exc
    kwargs: dict[str, Any] = {
        "device": args.device,
        "local_files_only": args.local_files_only,
    }
    if args.cache_folder is not None:
        kwargs["cache_folder"] = str(args.cache_folder.expanduser().resolve())
    return SentenceTransformer(args.model, **kwargs)


def extract_records(
    records: Sequence[TextRecord],
    encoder: Any,
    *,
    config: StyleLooConfig,
    document_batch_size: int,
    sentence_tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Extract an NPZ-ready array mapping; encoder injection keeps tests offline."""

    if not records:
        raise ValueError("input contains no records")
    if (
        isinstance(document_batch_size, bool)
        or not isinstance(document_batch_size, int)
        or document_batch_size < 1
    ):
        raise ValueError("document_batch_size must be a positive integer")
    blocks = []
    token_counts = []
    context_counts = []
    eligible_counts = []
    concentrations = []
    for start in tqdm(
        range(0, len(records), document_batch_size),
        desc="StyleDistance LOO",
        unit="block",
    ):
        result = build_style_loo_waves(
            encoder,
            [record.text for record in records[start : start + document_batch_size]],
            config=config,
            sentence_tokenizer=sentence_tokenizer,
        )
        blocks.append(result.waves)
        token_counts.append(result.token_counts)
        context_counts.append(result.context_counts)
        eligible_counts.append(result.eligible_token_counts)
        concentrations.append(result.document_direction_concentrations)
        del result
        gc.collect()

    labels = [record.label for record in records]
    if any(label is None for label in labels) and not all(label is None for label in labels):
        raise ValueError("label must be present for every record or absent from every record")
    arrays: dict[str, np.ndarray] = {
        "waves": np.concatenate(blocks).astype(np.float32, copy=False),
        "channel_names": np.asarray(STYLE_CHANNEL_NAMES, dtype=np.str_),
        "ids": np.asarray([record.id for record in records], dtype=np.str_),
        "models": np.asarray([record.model for record in records], dtype=np.str_),
        "sources": np.asarray([record.source for record in records], dtype=np.str_),
        "group_ids": np.asarray([record.group_id for record in records], dtype=np.str_),
        "attacks": np.asarray([record.attack for record in records], dtype=np.str_),
        "fingerprints": np.asarray(
            [text_fingerprint(record.text) for record in records], dtype=np.str_
        ),
        "style_token_counts": np.concatenate(token_counts),
        "style_context_counts": np.concatenate(context_counts),
        "style_eligible_token_counts": np.concatenate(eligible_counts),
        "style_direction_concentrations": np.concatenate(concentrations),
    }
    if labels[0] is not None:
        arrays["labels"] = np.asarray(labels, dtype=np.int64)
    return arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styleslip",
        description="Build standalone StyleDistance leave-one-out waves.",
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL or CSV input")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz archive")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "csv"), default="auto")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-folder", type=Path)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only an already cached StyleDistance model (default: true)",
    )
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--wave-length", type=_positive_int, default=256)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--context-tokens", type=_positive_int, default=128)
    parser.add_argument("--max-tokens", type=_positive_int)
    parser.add_argument("--document-batch-size", type=_positive_int, default=4)
    parser.add_argument("--encode-batch-size", type=_positive_int, default=256)
    parser.add_argument("--embedding-output-chunk-size", type=_positive_int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    _add_nltk_data_path(args.nltk_data or _discover_nltk_data_path())
    records = list(
        iter_records(
            args.input,
            input_format=args.input_format,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    )
    config = StyleLooConfig(
        output_length=args.wave_length,
        sigma=args.sigma,
        context_tokens=args.context_tokens,
        max_tokens=args.max_tokens,
        encode_batch_size=args.encode_batch_size,
        embedding_output_chunk_size=args.embedding_output_chunk_size,
    )
    encoder = _load_encoder(args)
    arrays = extract_records(
        records,
        encoder,
        config=config,
        document_batch_size=args.document_batch_size,
    )
    extraction_config = {
        "feature_set": "style-loo",
        "model": args.model,
        "document_scope": "full" if args.max_tokens is None else "prefix",
        "channels": list(STYLE_CHANNEL_NAMES),
        "style_loo": config.to_dict(),
    }
    arrays["config_json"] = np.asarray(json.dumps(extraction_config, ensure_ascii=False))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary = {
        "status": "completed",
        "input": str(args.input.expanduser().resolve()),
        "output": str(output),
        "documents": len(records),
        "shape": list(arrays["waves"].shape),
        "labeled": "labels" in arrays,
        "channels": list(STYLE_CHANNEL_NAMES),
        "config": config.to_dict(),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("Saved %s", output)
    return 0


__all__ = ["build_parser", "extract_records", "main"]
