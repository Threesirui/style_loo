"""Run paper-traceable classical baselines on StyleSlip dataset protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn

from styleslip.datasets import (
    ExperimentCase,
    InvalidRecordReport,
    SourceSpec,
    discover_cases,
    load_source,
    source_identity,
)
from styleslip.io import TextRecord

from .common import (
    apply_overlap_policy,
    group_split_records,
    labels,
    metrics,
    overlap_with_reference,
    select_threshold,
    texts,
    write_json,
    write_predictions,
)
from .stylometric_lr import StylometricLR
from .tfidf_svm import PAN25_COMMIT, TfidfSVM


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHODS = ("enhanced_tfidf_svm", "stylometric_lr", "pan25_tfidf_svm")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seed: int
    samples_per_class: int | None
    ood_samples_per_class: int | None
    deduplicate: bool
    overlap_policy: str
    invalid_record_policy: str
    val_size: float
    test_size: float
    threshold_mode: str
    c_grid: tuple[float, ...]
    stylo_profile: str
    word_max_features: int | None
    char_max_features: int | None


def _scenario(dataset: str, requested: str) -> str:
    if requested == "auto":
        return {"m4": "both", "deepfake": "all", "raid": "both", "all": "all"}[dataset]
    normalized = requested.strip().casefold().replace("-", "_")
    return {
        "single_language": "monolingual",
        "bilingual": "multilingual",
        "multi_language": "multilingual",
        "cross_domains_corss_models": "cross_domains_cross_models",
        "no_attack": "clean",
        "without_attack": "clean",
        "attack": "attacked",
        "with_attack": "attacked",
    }.get(normalized, normalized)


def _selected_cases(args: argparse.Namespace) -> list[ExperimentCase]:
    scenario = _scenario(args.dataset, args.scenario)
    if args.dataset == "all":
        return [
            *discover_cases(args.data_root, dataset="m4", scenario="both"),
            *discover_cases(args.data_root, dataset="deepfake", scenario="all", case_filter=args.case),
            *discover_cases(args.data_root, dataset="raid", scenario="both"),
        ]
    return discover_cases(
        args.data_root,
        dataset=args.dataset,
        scenario=scenario,
        case_filter=args.case,
    )


def _load(
    source: SourceSpec, args: argparse.Namespace
) -> tuple[list[TextRecord], InvalidRecordReport]:
    offset = {"train": 0, "source": 0, "validation": 1, "test": 2, "ood": 3}[source.split]
    limit = args.ood_samples_per_class if source.split == "ood" else args.samples_per_class
    report = InvalidRecordReport()
    records = load_source(
        source,
        samples_per_class=limit,
        seed=args.seed + offset,
        deduplicate=args.deduplicate,
        invalid_policy=args.invalid_record_policy,
        invalid_report=report,
    )
    return records, report


def _prepare_splits(
    case: ExperimentCase, args: argparse.Namespace
) -> tuple[dict[str, list[TextRecord]], dict[str, object]]:
    source_records, train_invalid = _load(case.train, args)
    invalid_records = {case.train.split: train_invalid.to_dict()}
    if case.protocol == "internal-group-split":
        train, validation, test, indices = group_split_records(
            source_records,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        details: dict[str, object] = {
            "strategy": "group-disjoint-source-id",
            "indices": indices,
            "overlap_counts": {"train": 0, "validation": 0, "test": 0},
        }
    else:
        assert case.validation is not None and case.test is not None
        validation_records, validation_invalid = _load(case.validation, args)
        test_records, test_invalid = _load(case.test, args)
        invalid_records["validation"] = validation_invalid.to_dict()
        invalid_records["test"] = test_invalid.to_dict()
        train, validation, test, overlap_counts = apply_overlap_policy(
            source_records,
            validation_records,
            test_records,
            policy=args.overlap_policy,
        )
        details = {"strategy": "dataset-authored", "overlap_counts": overlap_counts}
    details["invalid_records"] = invalid_records
    splits = {"train": train, "validation": validation, "test": test}
    if args.evaluate_ood and case.ood is not None:
        splits["ood"], ood_invalid = _load(case.ood, args)
        invalid_records["ood"] = ood_invalid.to_dict()
        details["ood_overlap_with_train_validation_test"] = overlap_with_reference(
            [*train, *validation, *test], splits["ood"]
        )
    return splits, details


def _make_model(method: str, args: argparse.Namespace):
    if method == "stylometric_lr":
        return StylometricLR(profile=args.stylo_profile)
    if method == "pan25_tfidf_svm":
        return TfidfSVM(profile="pan25")
    if method == "enhanced_tfidf_svm":
        return TfidfSVM(
            profile="enhanced",
            word_max_features=args.word_max_features,
            char_max_features=args.char_max_features,
        )
    raise ValueError(f"unknown baseline method: {method}")


def _fit_model(model, method: str, splits: dict[str, list[TextRecord]], args: argparse.Namespace) -> None:
    train_texts, train_labels = texts(splits["train"]), labels(splits["train"])
    if isinstance(model, TfidfSVM):
        model.fit(
            train_texts,
            train_labels,
            validation_texts=texts(splits["validation"]),
            validation_labels=labels(splits["validation"]),
            c_grid=args.c_grid if method == "enhanced_tfidf_svm" else None,
        )
    else:
        model.fit(train_texts, train_labels)


def _source_manifest(case: ExperimentCase, include_ood: bool) -> dict[str, object]:
    result = {"train": source_identity(case.train)}
    if case.validation is not None:
        result["validation"] = source_identity(case.validation)
    if case.test is not None:
        result["test"] = source_identity(case.test)
    if include_ood and case.ood is not None:
        result["ood"] = source_identity(case.ood)
    return result


def _run_hash(
    case: ExperimentCase, method: str, config: EvaluationConfig, sources: dict[str, object]
) -> str:
    value = {
        "format_version": 1,
        "case": case.slug,
        "method": method,
        "config": asdict(config),
        "sources": sources,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _provenance(method: str) -> dict[str, object]:
    if method in {"enhanced_tfidf_svm", "pan25_tfidf_svm"}:
        return {
            "upstream": "pan-webis-de/pan25-generative-ai-authorship-verification",
            "url": "https://github.com/pan-webis-de/pan25-generative-ai-authorship-verification",
            "commit": PAN25_COMMIT,
            "license": "Apache-2.0",
            "upstream_file": "pan25_genai_detection/baselines/tfidf.py",
            "relationship": (
                "configuration-compatible reimplementation"
                if method == "pan25_tfidf_svm"
                else "documented extension of the official word-TF-IDF + LinearSVC baseline"
            ),
        }
    return {
        "paper": "BertAA: BERT fine-tuning for Authorship Attribution",
        "paper_url": "https://aclanthology.org/2020.icon-main.16/",
        "official_notebook": "https://colab.research.google.com/drive/1m4anWkkb8tz3fKvzJFytygBkqCTdZ8bo",
        "notebook_sha256": "29dc3bd622306111ab72a33dd12fd20a920febd3b5662297d5fdc5fd8f744a10",
        "downstream_paper": "Team Innovative at SemEval-2024 Task 8",
        "downstream_paper_url": "https://aclanthology.org/2024.semeval-1.171/",
        "license": "not stated in the public notebook",
        "relationship": "clean-room behavioral reimplementation; no source copied or vendored",
    }


def run_method(
    case: ExperimentCase,
    method: str,
    splits: dict[str, list[TextRecord]],
    split_details: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    config = EvaluationConfig(
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        ood_samples_per_class=args.ood_samples_per_class,
        deduplicate=args.deduplicate,
        overlap_policy=args.overlap_policy,
        invalid_record_policy=args.invalid_record_policy,
        val_size=args.val_size,
        test_size=args.test_size,
        threshold_mode=args.threshold_mode,
        c_grid=tuple(args.c_grid),
        stylo_profile=args.stylo_profile,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )
    sources = _source_manifest(case, args.evaluate_ood)
    run_hash = _run_hash(case, method, config, sources)
    run_dir = (
        args.output_dir.expanduser().resolve()
        / method
        / case.dataset
        / case.scenario
        / case.name
        / run_hash[:12]
    )
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "experiment_manifest.json"
    model_path = run_dir / "model.pkl"
    if args.reuse_results and not args.refresh_results and all(
        path.is_file() for path in (metrics_path, manifest_path, model_path)
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and manifest.get("run_hash") == run_hash:
            return {
                "method": method,
                "case": case.slug,
                "run_dir": str(run_dir),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "reused": True,
            }

    run_dir.mkdir(parents=True, exist_ok=True)
    model = _make_model(method, args)
    _fit_model(model, method, splits, args)
    scores = {name: model.predict_score(texts(records)) for name, records in splits.items()}
    threshold = (
        select_threshold(labels(splits["validation"]), scores["validation"])
        if args.threshold_mode == "validation_f1"
        else 0.5
    )
    result_metrics = {
        "threshold_selection": {
            "mode": args.threshold_mode,
            "selected_on": "validation" if args.threshold_mode == "validation_f1" else None,
            "value": threshold,
        },
        **{
            name: metrics(labels(records), scores[name], threshold)
            for name, records in splits.items()
            if name != "train"
        },
        "fixed_0_5": {
            name: metrics(labels(records), scores[name], 0.5)
            for name, records in splits.items()
            if name != "train"
        },
    }
    for name, records in splits.items():
        if name != "train":
            write_predictions(run_dir / f"{name}_predictions.csv", records, scores[name], threshold, name)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    write_json(metrics_path, result_metrics)
    write_json(
        manifest_path,
        {
            "status": "completed",
            "run_hash": run_hash,
            "case": case.slug,
            "protocol": case.protocol,
            "method": method,
            "model_specification": model.specification(),
            "evaluation_config": asdict(config),
            "split_details": split_details,
            "split_sizes": {name: len(records) for name, records in splits.items()},
            "fit_scope": {
                "vectorizers_and_model": ["train"],
                "hyperparameters": ["validation"] if method == "enhanced_tfidf_svm" else [],
                "threshold": ["validation"] if args.threshold_mode == "validation_f1" else [],
                "test_or_ood_used_for_fit": False,
            },
            "sources": sources,
            "provenance": _provenance(method),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "blind_test": str(case.blind_test.resolve()) if case.blind_test else None,
            "blind_test_evaluated": False,
        },
    )
    return {
        "method": method,
        "case": case.slug,
        "run_dir": str(run_dir),
        "metrics": result_metrics,
        "reused": False,
    }


def _write_summary(output_dir: Path, dataset: str, results: Sequence[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"results_{dataset}.json", list(results))
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {
            "method": result["method"],
            "case": result["case"],
            "run_dir": result["run_dir"],
            "reused": result["reused"],
        }
        result_metrics = result["metrics"]
        assert isinstance(result_metrics, dict)
        for split in ("validation", "test", "ood"):
            split_metrics = result_metrics.get(split)
            if isinstance(split_metrics, dict):
                for key in ("samples", "auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier", "threshold"):
                    row[f"{split}_{key}"] = split_metrics[key]
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / f"results_{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _c_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("C grid must be comma-separated numbers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("C grid values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styleslip-baselines",
        description="Run Enhanced TF-IDF-SVM and paper/source-traceable classical baselines.",
    )
    parser.add_argument("--dataset", default="m4", choices=("m4", "deepfake", "raid", "all"))
    parser.add_argument("--scenario", default="auto")
    parser.add_argument("--case", help="Substring filter for Deepfake case directories")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines")
    parser.add_argument("--samples-per-class", type=_positive_int)
    parser.add_argument("--ood-samples-per-class", type=_positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-policy", choices=("error", "drop", "allow"), default="error")
    parser.add_argument(
        "--invalid-record-policy",
        choices=("skip", "error"),
        default="skip",
        help="Skip malformed individual records with a manifest audit, or fail immediately",
    )
    parser.add_argument("--evaluate-ood", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-size", type=_fraction, default=0.15)
    parser.add_argument("--test-size", type=_fraction, default=0.15)
    parser.add_argument("--threshold-mode", choices=("validation_f1", "fixed_0_5"), default="validation_f1")
    parser.add_argument("--c-grid", type=_c_grid, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--word-max-features", type=_positive_int, default=100_000)
    parser.add_argument("--char-max-features", type=_positive_int, default=200_000)
    parser.add_argument("--stylo-profile", choices=("bertaa_code", "paper_full"), default="bertaa_code")
    parser.add_argument("--reuse-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-results", action="store_true")
    parser.add_argument("--list", action="store_true", help="List cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show planned case/method pairs and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.val_size + args.test_size >= 1:
        parser.error("--val-size and --test-size must sum to less than 1")
    methods = args.method or ["enhanced_tfidf_svm", "stylometric_lr"]
    cases = _selected_cases(args)
    plan = [
        {
            "case": case.slug,
            "protocol": case.protocol,
            "method": method,
            "train": str(case.train.path.resolve()),
            "validation": str(case.validation.path.resolve()) if case.validation else "internal group split",
            "test": str(case.test.path.resolve()) if case.test else "internal group split",
            "ood": str(case.ood.path.resolve()) if args.evaluate_ood and case.ood else None,
        }
        for case in cases
        for method in methods
    ]
    if args.list or args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    results: list[dict[str, object]] = []
    for case in cases:
        splits, split_details = _prepare_splits(case, args)
        for method in methods:
            print(f"Running {method}: {case.slug}", flush=True)
            results.append(run_method(case, method, splits, split_details, args))
    _write_summary(args.output_dir.expanduser().resolve(), args.dataset, results)
    return 0


__all__ = ["build_parser", "main", "run_method"]
