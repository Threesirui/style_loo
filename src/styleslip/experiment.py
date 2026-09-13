"""Unified M4, Deepfake, and RAID Style-LOO experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import nltk
import numpy as np
from tqdm.auto import tqdm

from .cache import FeatureCache, cache_key, cache_spec
from .datasets import ExperimentCase, SourceSpec, discover_cases, source_identity
from .model import ModelConfig
from .style_loo import StyleLooConfig
from .training import (
    TrainConfig,
    enforce_overlap_policy,
    group_split,
    load_archive,
    train_model,
)


LOGGER = logging.getLogger("styleslip.experiment")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT.parent / "Textwave" / "model" / "styledistance_synthetic_only"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _scenario(dataset: str, requested: str) -> str:
    if requested == "auto":
        return {"m4": "both", "deepfake": "all", "raid": "both", "all": "all"}[dataset]
    normalized = requested.strip().casefold().replace("-", "_")
    aliases = {
        "single_language": "monolingual",
        "bilingual": "multilingual",
        "multi_language": "multilingual",
        "cross_domains_corss_models": "cross_domains_cross_models",
        "no_attack": "clean",
        "without_attack": "clean",
        "attack": "attacked",
        "with_attack": "attacked",
    }
    return aliases.get(normalized, normalized)


def _add_nltk_path(path: Path) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved not in nltk.data.path:
        nltk.data.path.insert(0, resolved)


class EncoderProvider:
    def __init__(
        self,
        model: str,
        device: str,
        local_files_only: bool,
        cache_folder: Path | None,
    ) -> None:
        self.model = model
        self.device = device
        self.local_files_only = local_files_only
        self.cache_folder = cache_folder
        self._encoder: Any = None

    def __call__(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for experiments") from exc
            kwargs: dict[str, Any] = {
                "device": self.device,
                "local_files_only": self.local_files_only,
            }
            if self.cache_folder is not None:
                kwargs["cache_folder"] = str(self.cache_folder.expanduser().resolve())
            LOGGER.info("Loading StyleDistance model: %s", self.model)
            self._encoder = SentenceTransformer(self.model, **kwargs)
        return self._encoder


def _sources(case: ExperimentCase, evaluate_ood: bool) -> list[SourceSpec]:
    values = [case.train]
    if case.validation is not None:
        values.append(case.validation)
    if case.test is not None:
        values.append(case.test)
    if evaluate_ood and case.ood is not None:
        values.append(case.ood)
    return values


def _split_seed(seed: int, split: str) -> int:
    return seed + {"train": 0, "source": 0, "validation": 1, "test": 2, "ood": 3}[split]


def _sample_limit(args: argparse.Namespace, source: SourceSpec) -> int | None:
    return args.ood_samples_per_class if source.split == "ood" else args.samples_per_class


def _prepare_case(
    case: ExperimentCase,
    args: argparse.Namespace,
    style_config: StyleLooConfig,
    cache: FeatureCache,
    encoder: EncoderProvider,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for source in _sources(case, args.evaluate_ood):
        result = cache.prepare(
            source,
            dataset=case.dataset,
            samples_per_class=_sample_limit(args, source),
            seed=_split_seed(args.seed, source.split),
            deduplicate=args.deduplicate,
            model=args.style_model,
            config=style_config,
            document_batch_size=args.style_document_batch_size,
            encoder_factory=encoder,
            refresh=args.refresh_feature_cache,
        )
        prepared[source.split] = {
            "path": str(result.path),
            "key": result.key,
            "reused": result.reused,
            "samples": result.samples,
        }
        LOGGER.info(
            "%s %s: %s (%d samples)",
            case.slug,
            source.split,
            "cache hit" if result.reused else "prepared",
            result.samples,
        )
    return prepared


def _dry_plan(
    cases: Sequence[ExperimentCase], args: argparse.Namespace, style_config: StyleLooConfig
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    cache_root = args.feature_cache_dir.expanduser().resolve()
    for case in cases:
        entries = []
        for source in _sources(case, args.evaluate_ood):
            spec = cache_spec(
                source,
                samples_per_class=_sample_limit(args, source),
                seed=_split_seed(args.seed, source.split),
                deduplicate=args.deduplicate,
                model=args.style_model,
                config=style_config,
            )
            key = cache_key(spec)
            entries.append(
                {
                    "split": source.split,
                    "input": str(source.path.resolve()),
                    "cache": str(cache_root / case.dataset / key[:2] / f"{key}.npz"),
                    "cache_exists": (
                        cache_root / case.dataset / key[:2] / f"{key}.npz"
                    ).is_file(),
                }
            )
        plans.append(
            {
                "case": case.slug,
                "protocol": case.protocol,
                "sources": entries,
                "blind_test": str(case.blind_test.resolve()) if case.blind_test else None,
                "blind_test_evaluated": False,
            }
        )
    return plans


def _run_key(
    case: ExperimentCase,
    prepared: dict[str, Any],
    train_config: TrainConfig,
    model_config: ModelConfig,
    overlap_policy: str,
) -> tuple[str, dict[str, Any]]:
    spec = {
        "format_version": 1,
        "case": case.slug,
        "protocol": case.protocol,
        "feature_cache_keys": {key: value["key"] for key, value in prepared.items()},
        "train": asdict(train_config),
        "model": model_config.to_dict(),
        "overlap_policy": overlap_policy,
    }
    key = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return key, spec


def _case_row(case: ExperimentCase, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": case.dataset,
        "scenario": case.scenario,
        "case": case.name,
        "protocol": case.protocol,
        "run_dir": str(run_dir),
        "blind_test": str(case.blind_test.resolve()) if case.blind_test else None,
        "blind_test_evaluated": False,
    }
    for scope in ("test", "ood"):
        values = metrics.get(scope)
        if values:
            for metric in ("samples", "auroc", "auprc", "accuracy", "precision", "recall", "f1", "brier"):
                row[f"{scope}_{metric}"] = values.get(metric)
    return row


def _write_summary(output_root: Path, dataset: str, rows: Sequence[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"results_{dataset}.json"
    csv_path = output_root / f"results_{dataset}.csv"
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    case: ExperimentCase,
    args: argparse.Namespace,
    style_config: StyleLooConfig,
    cache: FeatureCache,
    encoder: EncoderProvider,
) -> dict[str, Any] | None:
    prepared = _prepare_case(case, args, style_config, cache, encoder)
    if args.prepare_only:
        return None
    train_config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        require_cuda=args.require_cuda,
        amp=args.amp,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    model_config = ModelConfig(
        in_channels=3,
        channels=args.tcn_channels,
        depth=args.tcn_depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
    )
    run_key, run_spec = _run_key(
        case, prepared, train_config, model_config, args.overlap_policy
    )
    run_dir = args.output_dir.expanduser().resolve() / case.dataset / case.scenario / case.name / run_key[:12]
    manifest_path = run_dir / "experiment_manifest.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    required_results = [manifest_path, metrics_path, checkpoint_path]
    if case.protocol == "internal-group-split":
        required_results.append(run_dir / "split_indices.npz")
    if args.reuse_results and not args.refresh_results and all(
        path.is_file() for path in required_results
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and manifest.get("spec") == run_spec:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            LOGGER.info("%s: result cache hit", case.slug)
            return _case_row(case, run_dir, metrics)

    train = load_archive(Path(prepared[case.train.split]["path"]))
    split_details: dict[str, Any]
    if case.protocol == "predefined":
        assert case.validation is not None and case.test is not None
        validation = load_archive(Path(prepared["validation"]["path"]))
        test = load_archive(Path(prepared["test"]["path"]))
        train, validation, test, dropped = enforce_overlap_policy(
            train, validation, test, policy=args.overlap_policy
        )
        split_details = {"strategy": "dataset-authored", "overlap_dropped": dropped}
    else:
        train, validation, test, indices = group_split(
            train, val_size=args.val_size, test_size=args.test_size, seed=args.seed
        )
        split_details = {"strategy": "group-disjoint-source-id", "indices": indices}
    ood = load_archive(Path(prepared["ood"]["path"])) if "ood" in prepared else None

    run_dir.mkdir(parents=True, exist_ok=True)
    if case.protocol == "internal-group-split":
        np.savez_compressed(
            run_dir / "split_indices.npz",
            train=np.asarray(split_details["indices"]["train"], dtype=np.int64),
            validation=np.asarray(split_details["indices"]["validation"], dtype=np.int64),
            test=np.asarray(split_details["indices"]["test"], dtype=np.int64),
        )
    manifest_path.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "spec": run_spec,
                "prepared": prepared,
                "split_details": split_details,
                "blind_test": source_identity(
                    SourceSpec(case.blind_test, "raid", "test")
                ) if case.blind_test else None,
                "blind_test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = train_model(
        train,
        validation,
        test,
        output_dir=run_dir,
        train_config=train_config,
        model_config=model_config,
        ood=ood,
        show_progress=args.progress,
        show_batch_progress=args.batch_progress,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return _case_row(case, run_dir, metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styleslip-experiment",
        description="Run cached Style-LOO experiments on M4, Deepfake, or RAID.",
    )
    parser.add_argument("--dataset", default="deepfake", choices=("m4", "deepfake", "raid", "all"))
    parser.add_argument(
        "--scenario",
        default="unseen_domains",
        help=(
            "M4: monolingual/multilingual/both; Deepfake: cross_domains_cross_models/"
            "unseen_models/unseen_domains/all; RAID: clean/attacked/both"
        ),
    )
    parser.add_argument("--case", help="Substring filter for Deepfake case directories")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--feature-cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--list", action="store_true", help="List selected cases without processing data")
    parser.add_argument("--dry-run", action="store_true", help="Show inputs and cache targets")
    parser.add_argument("--prepare-only", default=False, action="store_true")
    parser.add_argument("--samples-per-class", type=_positive_int,default=1000, help="Number of samples per class for train/val/test")
    parser.add_argument("--ood-samples-per-class", type=_positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-policy", choices=("error", "drop", "allow"), default="allow", help="How to handle overlap between train/val/test/ood")
    parser.add_argument("--evaluate-ood", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--refresh-results", action="store_true")

    parser.add_argument(
        "--style-model",
        default="./model/styledistance_synthetic_only",
    )
    parser.add_argument("--style-device", default="cuda")
    parser.add_argument("--style-cache-folder", type=Path)
    parser.add_argument("--style-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wave-length", type=_positive_int, default=256)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--style-context-tokens", type=_positive_int, default=128)
    parser.add_argument("--style-max-tokens", type=_positive_int)
    parser.add_argument("--style-document-batch-size", type=_positive_int, default=4)
    parser.add_argument("--style-encode-batch-size", type=_positive_int, default=256)
    parser.add_argument("--style-embedding-output-chunk-size", type=_positive_int, default=2048)
    parser.add_argument("--nltk-data", type=Path, default=PROJECT_ROOT / ".nltk_data")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live epoch progress bar (default: true)",
    )
    parser.add_argument(
        "--batch-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also show a nested batch progress bar",
    )
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--patience", type=_positive_int, default=12)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=_fraction, default=0.15)
    parser.add_argument("--test-size", type=_fraction, default=0.15)
    parser.add_argument("--tcn-channels", type=_positive_int, default=64)
    parser.add_argument("--tcn-depth", type=_positive_int, default=3)
    parser.add_argument("--kernel-size", type=_positive_int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--classifier-hidden", type=_positive_int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.val_size + args.test_size >= 1.0:
        parser.error("--val-size and --test-size must sum to less than 1")
    if args.kernel_size < 3 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be odd and at least 3")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0,1)")
    selected_scenario = _scenario(args.dataset, args.scenario)
    if args.dataset == "all":
        cases = [
            *discover_cases(args.data_root, dataset="m4", scenario="both"),
            *discover_cases(
                args.data_root, dataset="deepfake", scenario="all", case_filter=args.case
            ),
            *discover_cases(args.data_root, dataset="raid", scenario="both"),
        ]
    else:
        cases = discover_cases(
            args.data_root,
            dataset=args.dataset,
            scenario=selected_scenario,
            case_filter=args.case,
        )
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "case": case.slug,
                        "protocol": case.protocol,
                        "train": str(case.train.path.resolve()),
                        "validation": str(case.validation.path.resolve()) if case.validation else None,
                        "test": str(case.test.path.resolve()) if case.test else None,
                        "ood": str(case.ood.path.resolve()) if case.ood else None,
                        "blind_test": str(case.blind_test.resolve()) if case.blind_test else None,
                    }
                    for case in cases
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    style_config = StyleLooConfig(
        output_length=args.wave_length,
        sigma=args.sigma,
        context_tokens=args.style_context_tokens,
        max_tokens=args.style_max_tokens,
        encode_batch_size=args.style_encode_batch_size,
        embedding_output_chunk_size=args.style_embedding_output_chunk_size,
    )
    if args.dry_run:
        print(json.dumps(_dry_plan(cases, args, style_config), ensure_ascii=False, indent=2))
        return 0
    _add_nltk_path(args.nltk_data)
    cache = FeatureCache(args.feature_cache_dir)
    encoder = EncoderProvider(
        args.style_model,
        args.style_device,
        args.style_local_files_only,
        args.style_cache_folder,
    )
    rows = []
    case_progress = tqdm(
        cases,
        desc="Experiment cases",
        unit="case",
        dynamic_ncols=True,
        disable=not args.progress,
    )
    for case in case_progress:
        if args.progress:
            case_progress.set_postfix_str(case.slug)
        LOGGER.info("Running %s", case.slug)
        row = run_case(case, args, style_config, cache, encoder)
        if row is not None:
            rows.append(row)
    if rows:
        _write_summary(args.output_dir.expanduser().resolve(), args.dataset, rows)
    return 0


__all__ = ["build_parser", "main", "run_case"]
