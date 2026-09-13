"""Training, overlap, and RAID group-split tests."""

from __future__ import annotations

import json

import numpy as np

from styleslip.model import ModelConfig
from styleslip.training import (
    ArchiveData,
    TrainConfig,
    enforce_overlap_policy,
    group_split,
    train_model,
)


def make_data(samples: int, seed: int = 1) -> ArchiveData:
    rng = np.random.default_rng(seed)
    labels = np.arange(samples, dtype=np.int64) % 2
    waves = rng.normal(0, 0.2, size=(samples, 3, 16)).astype(np.float32)
    waves += labels[:, None, None].astype(np.float32) * 0.8
    metadata = {
        "ids": np.asarray([f"id-{index}" for index in range(samples)]),
        "models": np.asarray(["human" if label == 0 else "gpt" for label in labels]),
        "sources": np.asarray(["news"] * samples),
        "group_ids": np.asarray([f"group-{index // 2}" for index in range(samples)]),
        "fingerprints": np.asarray([f"fp-{seed}-{index}" for index in range(samples)]),
    }
    return ArchiveData(waves, labels, metadata)


def test_group_split_is_disjoint_and_complete() -> None:
    data = make_data(80)
    train, validation, test, _ = group_split(data, val_size=0.2, test_size=0.2, seed=42)
    group_sets = [set(item.metadata["group_ids"].tolist()) for item in (train, validation, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert len(train.labels) + len(validation.labels) + len(test.labels) == 80


def test_overlap_policy_can_drop_later_duplicates() -> None:
    train = make_data(12, 1)
    validation = make_data(12, 2)
    test = make_data(12, 3)
    validation.metadata["fingerprints"][0] = train.metadata["fingerprints"][0]
    reduced = enforce_overlap_policy(train, validation, test, policy="drop")
    assert len(reduced[1].labels) == 11
    assert reduced[3]["validation"] == 1


def test_small_cpu_training_writes_model_and_metrics(tmp_path) -> None:
    metrics = train_model(
        make_data(24, 1),
        make_data(12, 2),
        make_data(12, 3),
        output_dir=tmp_path,
        train_config=TrainConfig(
            seed=7,
            epochs=2,
            patience=2,
            batch_size=6,
            device="cpu",
            require_cuda=False,
            amp=False,
        ),
        model_config=ModelConfig(channels=8, depth=1, classifier_hidden=8),
        ood=make_data(12, 4),
        show_progress=False,
    )
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "training_history.csv").is_file()
    assert (tmp_path / "training_status.json").is_file()
    status = json.loads((tmp_path / "training_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["completed_epochs"] == 2
    assert metrics["test"]["samples"] == 12
    assert metrics["ood"]["samples"] == 12
