"""Content-addressed feature cache shared by all experiment runs."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cli import extract_records
from .datasets import SourceSpec, load_source, source_identity
from .style_loo import STYLE_CHANNEL_NAMES, StyleLooConfig


LOGGER = logging.getLogger("styleslip.cache")


@dataclass(frozen=True, slots=True)
class CacheResult:
    path: Path
    key: str
    reused: bool
    samples: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _model_identity(model: str) -> dict[str, Any]:
    candidate = Path(model).expanduser()
    if not candidate.exists():
        return {"name": model}
    resolved = candidate.resolve()
    identity: dict[str, Any] = {"path": str(resolved)}
    for filename in ("model.safetensors", "pytorch_model.bin", "config.json", "modules.json"):
        file = resolved / filename if resolved.is_dir() else resolved
        if file.is_file():
            stat = file.stat()
            identity[filename] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if not resolved.is_dir():
                break
    return identity


def cache_spec(
    source: SourceSpec,
    *,
    samples_per_class: int | None,
    seed: int,
    deduplicate: bool,
    model: str,
    config: StyleLooConfig,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source": source_identity(source),
        "selection": {
            "samples_per_class": samples_per_class,
            "seed": seed,
            "deduplicate": deduplicate,
        },
        "model": _model_identity(model),
        "style_loo": config.to_dict(),
        "channels": list(STYLE_CHANNEL_NAMES),
    }


def cache_key(spec: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()


def _valid_archive(path: Path, manifest: Path, spec: dict[str, Any]) -> tuple[bool, int]:
    if not path.is_file() or not manifest.is_file():
        return False, 0
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("status") != "completed" or saved.get("spec") != spec:
            return False, 0
        with np.load(path, allow_pickle=False) as archive:
            required = {"waves", "labels", "channel_names", "fingerprints", "config_json"}
            if not required.issubset(archive.files):
                return False, 0
            waves = np.asarray(archive["waves"])
            labels = np.asarray(archive["labels"])
            names = tuple(np.asarray(archive["channel_names"]).astype(str).tolist())
            if waves.ndim != 3 or waves.shape[1] != 3 or len(labels) != len(waves):
                return False, 0
            if names != STYLE_CHANNEL_NAMES or not np.all(np.isfinite(waves)):
                return False, 0
            return True, int(len(waves))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False, 0


class FeatureCache:
    """Build each sampled source/configuration once and reuse its NPZ archive."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def prepare(
        self,
        source: SourceSpec,
        *,
        dataset: str,
        samples_per_class: int | None,
        seed: int,
        deduplicate: bool,
        model: str,
        config: StyleLooConfig,
        document_batch_size: int,
        encoder_factory: Callable[[], Any],
        refresh: bool = False,
    ) -> CacheResult:
        spec = cache_spec(
            source,
            samples_per_class=samples_per_class,
            seed=seed,
            deduplicate=deduplicate,
            model=model,
            config=config,
        )
        key = cache_key(spec)
        directory = self.root / dataset / key[:2]
        archive_path = directory / f"{key}.npz"
        manifest_path = directory / f"{key}.json"
        valid, samples = _valid_archive(archive_path, manifest_path, spec)
        if valid and not refresh:
            return CacheResult(archive_path, key, True, samples)

        LOGGER.info(
            "Feature cache miss; sampling %s (up to %s records per class)",
            source.path,
            samples_per_class if samples_per_class is not None else "all",
        )
        records = load_source(
            source,
            samples_per_class=samples_per_class,
            seed=seed,
            deduplicate=deduplicate,
        )
        LOGGER.info("Selected %d records from %s", len(records), source.path.name)
        encoder = encoder_factory()
        arrays = extract_records(
            records,
            encoder,
            config=config,
            document_batch_size=document_batch_size,
        )
        output_samples = int(len(arrays["waves"]))
        extraction_config = {
            "feature_set": "style-loo",
            "channels": list(STYLE_CHANNEL_NAMES),
            "cache_key": key,
            "cache_spec": spec,
        }
        arrays["config_json"] = np.asarray(
            json.dumps(extraction_config, ensure_ascii=False, sort_keys=True)
        )
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"status": "building", "spec": spec}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".npz", prefix=f"{key}.", dir=directory, delete=False
            ) as handle:
                temporary = Path(handle.name)
            np.savez_compressed(temporary, **arrays)
            temporary.replace(archive_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "spec": spec,
                        "archive": str(archive_path),
                        "samples": output_samples,
                        "selected_samples": len(records),
                        "skipped_samples": len(records) - output_samples,
                        "shape": list(arrays["waves"].shape),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return CacheResult(archive_path, key, False, output_samples)


__all__ = ["CacheResult", "FeatureCache", "cache_key", "cache_spec"]
