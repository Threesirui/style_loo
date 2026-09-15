"""Data splitting, evaluation, and artifact helpers for classical baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from styleslip.io import TextRecord, text_fingerprint


def labels(records: Sequence[TextRecord]) -> np.ndarray:
    values = np.asarray([record.label for record in records], dtype=np.int64)
    if set(np.unique(values).tolist()) != {0, 1}:
        raise ValueError("every evaluation split must contain both binary classes")
    return values


def texts(records: Sequence[TextRecord]) -> list[str]:
    return [record.text for record in records]


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("detector scores must all be finite")
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "samples": int(len(y_true)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # Brier score is defined for probability predictions. Binoculars emits
        # an unbounded ratio score, so it is intentionally unavailable there.
        "brier": (
            float(brier_score_loss(y_true, scores))
            if np.all((scores >= 0.0) & (scores <= 1.0))
            else None
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def group_split_records(
    records: Sequence[TextRecord], *, val_size: float, test_size: float, seed: int
) -> tuple[list[TextRecord], list[TextRecord], list[TextRecord], dict[str, list[int]]]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    y = labels(records)
    groups = np.asarray([record.group_id for record in records])
    if len(set(groups.tolist())) < 3:
        raise ValueError("group split requires at least three distinct group_ids")
    indices = np.arange(len(records))
    for attempt in range(100):
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_val, test = next(outer.split(indices, y, groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size / (1.0 - test_size),
            random_state=seed + 10_000 + attempt,
        )
        train_rel, validation_rel = next(
            inner.split(train_val, y[train_val], groups[train_val])
        )
        train = train_val[train_rel]
        validation = train_val[validation_rel]
        parts = (train, validation, test)
        if all(set(np.unique(y[part]).tolist()) == {0, 1} for part in parts):
            group_sets = [set(groups[part].tolist()) for part in parts]
            if all(group_sets[i].isdisjoint(group_sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
                subset = lambda part: [records[int(index)] for index in part]
                return (
                    subset(train),
                    subset(validation),
                    subset(test),
                    {
                        "train": train.astype(int).tolist(),
                        "validation": validation.astype(int).tolist(),
                        "test": test.astype(int).tolist(),
                    },
                )
    raise ValueError("could not create group-disjoint splits containing both classes")


def apply_overlap_policy(
    train: Sequence[TextRecord],
    validation: Sequence[TextRecord],
    test: Sequence[TextRecord],
    *,
    policy: str,
) -> tuple[list[TextRecord], list[TextRecord], list[TextRecord], dict[str, int]]:
    if policy not in {"error", "drop", "allow"}:
        raise ValueError("overlap policy must be error, drop, or allow")
    seen: set[str] = set()
    result: list[list[TextRecord]] = []
    overlap_counts: dict[str, int] = {}
    for split, records in (("train", train), ("validation", validation), ("test", test)):
        fingerprints = [text_fingerprint(record.text) for record in records]
        overlap = [fingerprint in seen for fingerprint in fingerprints]
        overlap_counts[split] = sum(overlap)
        if any(overlap) and policy == "error":
            raise ValueError(f"{sum(overlap)} normalized texts overlap an earlier split in {split}")
        kept = [record for record, duplicate in zip(records, overlap) if not duplicate or policy == "allow"]
        labels(kept)
        result.append(kept)
        seen.update(
            fingerprint
            for fingerprint, duplicate in zip(fingerprints, overlap)
            if not duplicate or policy == "allow"
        )
    return result[0], result[1], result[2], overlap_counts


def overlap_with_reference(reference: Iterable[TextRecord], target: Iterable[TextRecord]) -> int:
    known = {text_fingerprint(record.text) for record in reference}
    return sum(text_fingerprint(record.text) in known for record in target)


def write_predictions(
    path: Path,
    records: Sequence[TextRecord],
    scores: np.ndarray,
    threshold: float,
    split: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "id", "model", "source", "attack", "group_id", "label", "score", "prediction", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (record, score) in enumerate(zip(records, scores)):
            writer.writerow(
                {
                    "index": index,
                    "id": record.id,
                    "model": record.model,
                    "source": record.source,
                    "attack": record.attack,
                    "group_id": record.group_id,
                    "label": record.label,
                    "score": float(score),
                    "prediction": int(score >= threshold),
                    "split": split,
                }
            )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "apply_overlap_policy",
    "group_split_records",
    "labels",
    "metrics",
    "overlap_with_reference",
    "select_threshold",
    "texts",
    "write_json",
    "write_predictions",
]
