"""Train, validate, and evaluate a Style-LOO temporal classifier."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import ModelConfig, StyleWaveTCN


@dataclass(slots=True)
class ArchiveData:
    waves: np.ndarray
    labels: np.ndarray
    metadata: dict[str, np.ndarray]

    def subset(self, indices: np.ndarray) -> "ArchiveData":
        return ArchiveData(
            self.waves[indices],
            self.labels[indices],
            {key: values[indices] for key, values in self.metadata.items()},
        )


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    patience: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    require_cuda: bool = True
    amp: bool = True
    val_size: float = 0.15
    test_size: float = 0.15


def load_archive(path: Path) -> ArchiveData:
    with np.load(path, allow_pickle=False) as archive:
        required = {"waves", "labels"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: missing waves or labels")
        waves = np.asarray(archive["waves"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        metadata = {
            key: np.asarray(archive[key]).astype(str).reshape(-1)
            for key in ("ids", "models", "sources", "group_ids", "attacks", "fingerprints")
            if key in archive.files
        }
    if waves.ndim != 3 or waves.shape[0] != len(labels) or waves.shape[1] != 3:
        raise ValueError(f"{path}: expected waves [N,3,L] aligned with labels")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{path}: both binary classes are required")
    return ArchiveData(waves, labels, metadata)


def _both_classes(labels: np.ndarray) -> bool:
    return set(np.unique(labels).tolist()) == {0, 1}


def group_split(
    data: ArchiveData, *, val_size: float, test_size: float, seed: int
) -> tuple[ArchiveData, ArchiveData, ArchiveData, dict[str, list[int]]]:
    """Create deterministic group-disjoint train/validation/test partitions."""

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    groups = data.metadata.get("group_ids")
    if groups is None or len(set(groups.tolist())) < 3:
        raise ValueError("group split requires at least three distinct group_ids")
    indices = np.arange(len(data.labels))
    for attempt in range(100):
        outer = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed + attempt
        )
        train_val_pos, test_pos = next(outer.split(indices, data.labels, groups))
        relative_val = val_size / (1.0 - test_size)
        inner = GroupShuffleSplit(
            n_splits=1, test_size=relative_val, random_state=seed + 10_000 + attempt
        )
        train_rel, val_rel = next(
            inner.split(train_val_pos, data.labels[train_val_pos], groups[train_val_pos])
        )
        train_pos = train_val_pos[train_rel]
        val_pos = train_val_pos[val_rel]
        if all(_both_classes(data.labels[value]) for value in (train_pos, val_pos, test_pos)):
            split_groups = [set(groups[value].tolist()) for value in (train_pos, val_pos, test_pos)]
            if split_groups[0].isdisjoint(split_groups[1]) and split_groups[0].isdisjoint(
                split_groups[2]
            ) and split_groups[1].isdisjoint(split_groups[2]):
                return (
                    data.subset(train_pos),
                    data.subset(val_pos),
                    data.subset(test_pos),
                    {
                        "train": train_pos.astype(int).tolist(),
                        "validation": val_pos.astype(int).tolist(),
                        "test": test_pos.astype(int).tolist(),
                    },
                )
    raise ValueError("could not create group-disjoint splits containing both classes")


def enforce_overlap_policy(
    train: ArchiveData,
    validation: ArchiveData,
    test: ArchiveData,
    *,
    policy: str,
) -> tuple[ArchiveData, ArchiveData, ArchiveData, dict[str, int]]:
    """Detect normalized-text leakage across predefined dataset splits."""

    if policy not in {"error", "drop", "allow"}:
        raise ValueError("overlap policy must be error, drop, or allow")
    seen: set[str] = set()
    result: list[ArchiveData] = []
    dropped: dict[str, int] = {}
    for name, data in (("train", train), ("validation", validation), ("test", test)):
        fingerprints = data.metadata.get("fingerprints")
        if fingerprints is None:
            raise ValueError("fingerprints are required for overlap checks")
        overlap = np.asarray([value in seen for value in fingerprints], dtype=bool)
        count = int(overlap.sum())
        dropped[name] = count
        if count and policy == "error":
            raise ValueError(
                f"{count} normalized texts overlap an earlier predefined split in {name}; "
                "use --overlap-policy drop or allow"
            )
        selected = np.flatnonzero(~overlap) if policy == "drop" else np.arange(len(data.labels))
        reduced = data.subset(selected)
        if not _both_classes(reduced.labels):
            raise ValueError(f"overlap policy leaves {name} without both classes")
        result.append(reduced)
        seen.update(fingerprints[selected].tolist())
    return result[0], result[1], result[2], dropped


def _resolve_device(config: TrainConfig) -> torch.device:
    requested = config.device.casefold()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if config.require_cuda:
            raise RuntimeError("CUDA was required but is unavailable")
        requested = "cpu"
    return torch.device(requested)


def _loader(data: ArchiveData, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(data.waves, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(data.labels, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _predict(
    model: nn.Module,
    data: ArchiveData,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.inference_mode():
        for waves, labels in _loader(data, batch_size, False):
            waves = waves.to(device)
            labels = labels.to(device)
            logits = model(waves)
            losses.append(float(criterion(logits, labels).item()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities), sum(losses) / len(data.labels)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "samples": int(len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _write_predictions(
    path: Path,
    data: ArchiveData,
    probabilities: np.ndarray,
    threshold: float,
    split: str,
) -> None:
    keys = [key for key in ("ids", "models", "sources", "attacks", "group_ids") if key in data.metadata]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", *keys, "label", "probability", "prediction", "split"],
        )
        writer.writeheader()
        for index, (label, probability) in enumerate(zip(data.labels, probabilities)):
            writer.writerow(
                {
                    "index": index,
                    **{key: data.metadata[key][index] for key in keys},
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "split": split,
                }
            )


def train_model(
    train: ArchiveData,
    validation: ArchiveData,
    test: ArchiveData,
    *,
    output_dir: Path,
    train_config: TrainConfig,
    model_config: ModelConfig,
    ood: ArchiveData | None = None,
    show_progress: bool = True,
    show_batch_progress: bool = False,
) -> dict[str, Any]:
    """Fit one model, tune its threshold on validation, and report test/OOD."""

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = _resolve_device(train_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    means = train.waves.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    scales = train.waves.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    scales = np.maximum(scales, 1e-6)

    def normalized(data: ArchiveData) -> ArchiveData:
        values = (data.waves - means[None, :, None]) / scales[None, :, None]
        return ArchiveData(values.astype(np.float32), data.labels, data.metadata)

    train_n, validation_n, test_n = map(normalized, (train, validation, test))
    ood_n = normalized(ood) if ood is not None else None
    model = StyleWaveTCN(model_config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    use_amp = train_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    history_path = output_dir / "training_history.csv"
    status_path = output_dir / "training_status.json"
    epoch_progress = tqdm(
        range(1, train_config.epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with history_path.open("w", encoding="utf-8", newline="") as history_handle:
        history_writer = csv.DictWriter(
            history_handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "best_validation_loss",
                "best_epoch",
                "epochs_without_improvement",
                "epoch_seconds",
            ),
        )
        history_writer.writeheader()
        history_handle.flush()
        for epoch in epoch_progress:
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            batches = _loader(train_n, train_config.batch_size, True)
            batch_progress = tqdm(
                batches,
                desc=f"Epoch {epoch}/{train_config.epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=not show_batch_progress,
            )
            for batch_index, (waves, labels) in enumerate(batch_progress, start=1):
                waves = waves.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(waves)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * len(labels)
                if show_batch_progress:
                    batch_progress.set_postfix(
                        loss=f"{float(loss.item()):.4f}",
                        samples=min(batch_index * train_config.batch_size, len(train_n.labels)),
                    )
            train_loss = total_loss / len(train_n.labels)
            _, val_loss = _predict(
                model, validation_n, device=device, batch_size=train_config.batch_size
            )
            improved = val_loss < best_loss - 1e-4
            if improved:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            epoch_seconds = time.perf_counter() - epoch_started
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": stale,
                "epoch_seconds": epoch_seconds,
            }
            history.append(record)
            history_writer.writerow(record)
            history_handle.flush()
            status_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "device": str(device),
                        "current_epoch": epoch,
                        "maximum_epochs": train_config.epochs,
                        "progress_percent": 100.0 * epoch / train_config.epochs,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                        "best_validation_loss": best_loss,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": stale,
                        "patience": train_config.patience,
                        "early_stopping": stale >= train_config.patience,
                        "epoch_seconds": epoch_seconds,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            epoch_progress.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_loss:.4f}@{best_epoch}",
                patience=f"{stale}/{train_config.patience}",
            )
            if stale >= train_config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_probabilities, _ = _predict(
        model, validation_n, device=device, batch_size=train_config.batch_size
    )
    threshold = select_threshold(validation.labels, val_probabilities)
    test_probabilities, _ = _predict(
        model, test_n, device=device, batch_size=train_config.batch_size
    )
    metrics: dict[str, Any] = {
        "device": str(device),
        "best_epoch": best_epoch,
        "normalization": {"means": means.tolist(), "scales": scales.tolist()},
        "splits": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "validation": classification_metrics(validation.labels, val_probabilities, threshold),
        "test": classification_metrics(test.labels, test_probabilities, threshold),
        "history": history,
    }
    torch.save(
        {
            "state_dict": best_state,
            "model_config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "normalization_means": means,
            "normalization_scales": scales,
            "threshold": threshold,
        },
        output_dir / "best_model.pt",
    )
    _write_predictions(
        output_dir / "validation_predictions.csv",
        validation,
        val_probabilities,
        threshold,
        "validation",
    )
    _write_predictions(output_dir / "test_predictions.csv", test, test_probabilities, threshold, "test")
    if ood is not None and ood_n is not None:
        ood_probabilities, _ = _predict(
            model, ood_n, device=device, batch_size=train_config.batch_size
        )
        metrics["ood"] = classification_metrics(ood.labels, ood_probabilities, threshold)
        metrics["splits"]["ood"] = len(ood.labels)
        _write_predictions(output_dir / "ood_predictions.csv", ood, ood_probabilities, threshold, "ood")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "device": str(device),
                "completed_epochs": len(history),
                "maximum_epochs": train_config.epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "stopped_early": len(history) < train_config.epochs,
                "test_metrics": metrics["test"],
                **({"ood_metrics": metrics["ood"]} if "ood" in metrics else {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


__all__ = [
    "ArchiveData",
    "TrainConfig",
    "classification_metrics",
    "enforce_overlap_policy",
    "group_split",
    "load_archive",
    "train_model",
]
