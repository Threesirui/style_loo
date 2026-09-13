"""Dataset-aware discovery and bounded streaming loaders for experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from .io import TextRecord, text_fingerprint


DatasetName = Literal["m4", "deepfake", "raid"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    path: Path
    format: Literal["m4", "deepfake", "raid"]
    split: str


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    dataset: DatasetName
    scenario: str
    name: str
    protocol: Literal["predefined", "internal-group-split"]
    train: SourceSpec
    validation: SourceSpec | None = None
    test: SourceSpec | None = None
    ood: SourceSpec | None = None
    blind_test: Path | None = None

    @property
    def slug(self) -> str:
        return "/".join(part for part in (self.dataset, self.scenario, self.name) if part)


def _seed(seed: int, namespace: str) -> int:
    payload = f"StyleSlip:{seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _select_bounded(
    records: Iterator[TextRecord],
    *,
    samples_per_class: int | None,
    seed: int,
    deduplicate: bool,
) -> list[TextRecord]:
    if samples_per_class is not None and samples_per_class < 1:
        raise ValueError("samples_per_class must be positive or None")
    buckets: dict[int, list[TextRecord]] = {0: [], 1: []}
    seen_counts = {0: 0, 1: 0}
    rngs = {
        0: random.Random(_seed(seed, "human")),
        1: random.Random(_seed(seed, "ai")),
    }
    seen_fingerprints: set[str] = set()
    for record in records:
        if record.label not in (0, 1):
            raise ValueError("experiment records must have binary labels")
        if deduplicate:
            fingerprint = text_fingerprint(record.text)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
        label = int(record.label)
        seen_counts[label] += 1
        bucket = buckets[label]
        if samples_per_class is None:
            bucket.append(record)
        elif len(bucket) < samples_per_class:
            bucket.append(record)
        else:
            replacement = rngs[label].randrange(seen_counts[label])
            if replacement < samples_per_class:
                bucket[replacement] = record
    if not buckets[0] or not buckets[1]:
        raise ValueError("selected split must contain both human and AI records")
    selected = [*buckets[0], *buckets[1]]
    random.Random(_seed(seed, "final-order")).shuffle(selected)
    return selected


def _iter_m4(path: Path) -> Iterator[TextRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number}: invalid JSON") from exc
            text = raw.get("text")
            label = raw.get("label")
            model = raw.get("model")
            source = raw.get("source", raw.get("domain"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path} line {line_number}: text is empty")
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(f"{path} line {line_number}: label must be 0 or 1")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"{path} line {line_number}: model is empty")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{path} line {line_number}: source/domain is empty")
            record_id = str(raw.get("id", f"{path.stem}:{line_number}"))
            yield TextRecord(
                text=text,
                label=int(label),
                id=record_id,
                model=model.strip(),
                source=source.strip(),
                group_id=record_id,
            )


def _parse_bool(value: str, *, location: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{location}: is_human must be true/false")


def _iter_deepfake(path: Path) -> Iterator[TextRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label", "domain", "is_human", "model"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required Deepfake columns")
        for row_number, raw in enumerate(reader, start=2):
            location = f"{path} row {row_number}"
            text = (raw.get("text") or "").strip()
            domain = (raw.get("domain") or "").strip()
            is_human = _parse_bool(raw.get("is_human") or "", location=location)
            try:
                source_label = int((raw.get("label") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{location}: label must be 0 or 1") from exc
            if source_label != (1 if is_human else 0):
                raise ValueError(f"{location}: label conflicts with is_human")
            model = (raw.get("model") or "").strip()
            if not text or not domain or (not is_human and not model):
                raise ValueError(f"{location}: text/domain/model is incomplete")
            record_id = f"{path.parent.name}/{path.stem}:{row_number - 1}"
            yield TextRecord(
                text=text,
                label=0 if is_human else 1,
                id=record_id,
                model="human" if is_human else model,
                source=domain,
                group_id=record_id,
            )


def _iter_raid(path: Path) -> Iterator[TextRecord]:
    csv.field_size_limit(2**31 - 1)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "source_id", "model", "attack", "domain", "generation"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: file is not a labeled RAID split")
        for row_number, raw in enumerate(reader, start=2):
            text = raw.get("generation") or ""
            model = (raw.get("model") or "").strip()
            domain = (raw.get("domain") or "").strip()
            record_id = (raw.get("id") or "").strip()
            source_id = (raw.get("source_id") or record_id).strip()
            attack = (raw.get("attack") or "none").strip() or "none"
            if not text.strip() or not model or not domain or not record_id:
                raise ValueError(f"{path} row {row_number}: incomplete RAID record")
            is_human = model.casefold() == "human"
            yield TextRecord(
                text=text,
                label=0 if is_human else 1,
                id=record_id,
                model="human" if is_human else model,
                source=domain,
                group_id=source_id,
                attack=attack,
            )


def load_source(
    source: SourceSpec,
    *,
    samples_per_class: int | None,
    seed: int,
    deduplicate: bool = False,
) -> list[TextRecord]:
    """Load one split with deterministic per-class reservoir sampling."""

    path = source.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset split does not exist: {path}")
    iterator = {
        "m4": _iter_m4,
        "deepfake": _iter_deepfake,
        "raid": _iter_raid,
    }[source.format](path)
    return _select_bounded(
        iterator,
        samples_per_class=samples_per_class,
        seed=seed,
        deduplicate=deduplicate,
    )


def discover_m4(root: Path, scenario: str) -> list[ExperimentCase]:
    subtask = root / "SemEval2024-M4" / "SubtaskA"
    tracks = ("monolingual", "multilingual") if scenario in {"all", "both"} else (scenario,)
    cases: list[ExperimentCase] = []
    for track in tracks:
        if track not in {"monolingual", "multilingual"}:
            raise ValueError("M4 scenario must be monolingual, multilingual, or both")
        specs = {
            split: SourceSpec(subtask / f"subtaskA_{name}_{track}.jsonl", "m4", split)
            for split, name in (("train", "train"), ("validation", "dev"), ("test", "test"))
        }
        cases.append(
            ExperimentCase(
                dataset="m4",
                scenario=track,
                name="subtaskA",
                protocol="predefined",
                train=specs["train"],
                validation=specs["validation"],
                test=specs["test"],
            )
        )
    return cases


def discover_deepfake(root: Path, scenario: str, case_filter: str | None = None) -> list[ExperimentCase]:
    base = root / "Deepfake_For_Multil_Label"
    allowed = ("cross_domains_cross_models", "unseen_models", "unseen_domains")
    scenarios = allowed if scenario in {"all", "both"} else (scenario,)
    cases: list[ExperimentCase] = []
    for item in scenarios:
        if item not in allowed:
            raise ValueError(f"Deepfake scenario must be one of: {', '.join(allowed)}, all")
        scenario_dir = base / item
        directories = (
            [scenario_dir]
            if (scenario_dir / "train.csv").is_file()
            else sorted(path for path in scenario_dir.iterdir() if path.is_dir())
        )
        for directory in directories:
            name = "all" if directory == scenario_dir else directory.name
            if case_filter and case_filter.casefold() not in name.casefold():
                continue
            ood_files = sorted(directory.glob("test_ood*.csv"))
            if len(ood_files) > 1:
                raise ValueError(f"{directory}: multiple OOD files found")
            cases.append(
                ExperimentCase(
                    dataset="deepfake",
                    scenario=item,
                    name=name,
                    protocol="predefined",
                    train=SourceSpec(directory / "train.csv", "deepfake", "train"),
                    validation=SourceSpec(directory / "valid.csv", "deepfake", "validation"),
                    test=SourceSpec(directory / "test.csv", "deepfake", "test"),
                    ood=(SourceSpec(ood_files[0], "deepfake", "ood") if ood_files else None),
                )
            )
    if not cases:
        raise ValueError("no Deepfake cases matched the requested scenario/case")
    return cases


def discover_raid(root: Path, scenario: str) -> list[ExperimentCase]:
    base = root / "RAID"
    variants = ("clean", "attacked") if scenario in {"all", "both"} else (scenario,)
    cases: list[ExperimentCase] = []
    for variant in variants:
        if variant not in {"clean", "attacked"}:
            raise ValueError("RAID scenario must be clean, attacked, or both")
        suffix = "_none" if variant == "clean" else ""
        cases.append(
            ExperimentCase(
                dataset="raid",
                scenario=variant,
                name="train_internal_split",
                protocol="internal-group-split",
                train=SourceSpec(base / f"train{suffix}.csv", "raid", "source"),
                ood=SourceSpec(base / f"extra{suffix}.csv", "raid", "ood"),
                blind_test=base / f"test{suffix}.csv",
            )
        )
    return cases


def discover_cases(
    data_root: Path,
    *,
    dataset: str,
    scenario: str,
    case_filter: str | None = None,
) -> list[ExperimentCase]:
    root = data_root.expanduser().resolve()
    if dataset == "m4":
        return discover_m4(root, scenario)
    if dataset == "deepfake":
        return discover_deepfake(root, scenario, case_filter)
    if dataset == "raid":
        return discover_raid(root, scenario)
    if dataset == "all":
        return [
            *discover_m4(root, "both"),
            *discover_deepfake(root, "all", case_filter),
            *discover_raid(root, "both"),
        ]
    raise ValueError("dataset must be m4, deepfake, raid, or all")


def source_identity(source: SourceSpec) -> dict[str, Any]:
    path = source.path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "format": source.format,
        "split": source.split,
    }


__all__ = [
    "ExperimentCase",
    "SourceSpec",
    "discover_cases",
    "load_source",
    "source_identity",
]
