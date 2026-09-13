"""Feature cache reuse tests."""

from __future__ import annotations

import json
from pathlib import Path

import nltk

from styleslip.cache import FeatureCache
from styleslip.datasets import SourceSpec
from styleslip.style_loo import StyleLooConfig
from test_style_loo import FakeStyleEncoder


def test_identical_request_reuses_feature_archive(tmp_path) -> None:
    nltk_root = Path(__file__).resolve().parents[1] / ".nltk_data"
    nltk.data.path.insert(0, str(nltk_root))
    source = tmp_path / "sample.jsonl"
    rows = [
        {"id": "h1", "text": "Human writers revise these words.", "label": 0, "model": "human", "source": "news"},
        {"id": "h2", "text": "People carefully edit this sentence.", "label": 0, "model": "human", "source": "news"},
        {"id": "a1", "text": "Generated systems produce these words.", "label": 1, "model": "gpt", "source": "news"},
        {"id": "a2", "text": "Machine output forms this sentence.", "label": 1, "model": "gpt", "source": "news"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    calls = 0

    def factory() -> FakeStyleEncoder:
        nonlocal calls
        calls += 1
        return FakeStyleEncoder()

    cache = FeatureCache(tmp_path / "cache")
    kwargs = dict(
        dataset="m4",
        samples_per_class=2,
        seed=42,
        deduplicate=True,
        model="fake-model",
        config=StyleLooConfig(output_length=8),
        document_batch_size=2,
        encoder_factory=factory,
    )
    first = cache.prepare(SourceSpec(source, "m4", "train"), **kwargs)
    second = cache.prepare(SourceSpec(source, "m4", "train"), **kwargs)
    assert first.path == second.path
    assert not first.reused and second.reused
    assert first.samples == second.samples == 4
    assert calls == 1

