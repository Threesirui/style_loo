"""Tests for standalone inputs and NPZ-ready extraction."""

from __future__ import annotations

import json

import numpy as np

from styleslip.cli import extract_records
from styleslip.io import TextRecord, iter_records
from styleslip.style_loo import StyleLooConfig
from test_style_loo import FakeStyleEncoder


def single_sentence(text: str) -> list[str]:
    return [text]


def test_jsonl_reader_and_extraction(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "h1", "text": "Human words are written here.", "label": 0}),
                json.dumps({"id": "a1", "text": "Generated words appear over here.", "label": 1}),
            ]
        ),
        encoding="utf-8",
    )
    records = list(iter_records(path))
    arrays = extract_records(
        records,
        FakeStyleEncoder(),
        config=StyleLooConfig(output_length=16, context_tokens=32),
        document_batch_size=2,
        sentence_tokenizer=single_sentence,
    )
    assert arrays["waves"].shape == (2, 3, 16)
    assert arrays["labels"].tolist() == [0, 1]
    assert arrays["ids"].tolist() == ["h1", "a1"]
    assert np.all(np.isfinite(arrays["waves"]))


def test_unlabeled_records_are_supported() -> None:
    arrays = extract_records(
        [TextRecord("Several useful words live here.")],
        FakeStyleEncoder(),
        config=StyleLooConfig(output_length=8),
        document_batch_size=1,
        sentence_tokenizer=single_sentence,
    )
    assert "labels" not in arrays
