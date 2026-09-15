"""Thin adapters around the authors' released baseline implementations.

The method implementations live unchanged under ``baselines/upstream``. This
module only translates StyleSlip's common split objects to their APIs and
orients every score so larger values mean "more likely machine-generated".
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = PROJECT_ROOT / "baselines" / "upstream"
SEMEVAL_ROOT = UPSTREAM_ROOT / "SemEval2024-task8-main"
SEMEVAL_SCRIPT = SEMEVAL_ROOT / "subtaskA" / "baseline" / "transformer_baseline.py"
FAST_DETECT_ROOT = UPSTREAM_ROOT / "fast-detect-gpt-main"
FAST_DETECT_SCRIPTS = FAST_DETECT_ROOT / "scripts"
BINOCULARS_ROOT = UPSTREAM_ROOT / "Binoculars-main"

SEMEVAL_COMMIT = "d8350c840bc505eaba06b4baf69993c2d18fef5e"
FAST_DETECT_COMMIT = "971b05202bac2bb504d60c0ac0812fea7a8f7c82"
BINOCULARS_COMMIT = "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8"
FAST_DETECT_PAIRS = {
    ("gpt-j-6B", "gpt-neo-2.7B"),
    ("gpt-neo-2.7B", "gpt-neo-2.7B"),
    ("falcon-7b", "falcon-7b-instruct"),
}


def _load_module(name: str, path: Path, *, package_dir: Path | None = None):
    if not path.is_file():
        raise FileNotFoundError(
            f"required official source is missing: {path}. "
            "Run baselines/fetch_upstream.ps1 first."
        )
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    kwargs = (
        {"submodule_search_locations": [str(package_dir)]}
        if package_dir is not None
        else {}
    )
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_cuda_capacity(required_gib: float, label: str) -> None:
    """Fail before a large download when an official GPU setup cannot fit."""
    import torch

    if not torch.cuda.is_available():
        return
    total_gib = max(
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(torch.cuda.device_count())
    )
    if total_gib + 0.25 < required_gib:
        raise RuntimeError(
            f"{label} needs about {required_gib:.1f} GiB on one GPU in its official "
            f"precision, but the largest visible GPU has {total_gib:.1f} GiB. "
            "Use hardware that fits the released configuration; changing model size "
            "or quantization is a separate variant, not a paper-faithful reproduction."
        )


def _require_binoculars_capacity() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    capacities = [
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(torch.cuda.device_count())
    ]
    fits = capacities[0] >= 30.0 if len(capacities) == 1 else min(capacities[:2]) >= 15.0
    if not fits:
        visible = ", ".join(f"GPU{i}={value:.1f} GiB" for i, value in enumerate(capacities))
        raise RuntimeError(
            "Binoculars' official loader places one bfloat16 Falcon-7B model on "
            "each of the first two GPUs, or both on one GPU. It requires roughly "
            f"15 GiB per model (visible: {visible}). Quantized or smaller pairs "
            "must be reported as variants, not the paper-faithful baseline."
        )


class SemEvalXLMRAdapter:
    """Use the SemEval-2024 Task 8 authors' Transformer baseline code."""

    requires_training = True
    native_threshold = 0.5

    def __init__(self, *, model_name: str, run_dir: Path, seed: int) -> None:
        self.model_name = model_name
        self.run_dir = run_dir
        self.seed = seed
        self.checkpoints = run_dir / "upstream_checkpoints"
        self.best_model = self.checkpoints / "best"
        self._tokenizer = None
        self._model = None

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: np.ndarray,
        *,
        validation_texts: Sequence[str],
        validation_labels: np.ndarray,
        show_progress: bool = False,
    ) -> "SemEvalXLMRAdapter":
        del show_progress
        try:
            import pandas as pd
            from transformers import set_seed
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "The official SemEval baseline requires pandas, datasets, evaluate, "
                "accelerate, torch, and transformers. Install .[neural-baselines]."
            ) from exc
        upstream = _load_module("styleslip_upstream_semeval_transformer", SEMEVAL_SCRIPT)
        # Transformers 5 renamed two API arguments. Keep the authors' function
        # and numerical settings intact while translating only those API names.
        if not getattr(upstream, "_styleslip_compat_patched", False):
            training_parameters = inspect.signature(upstream.TrainingArguments).parameters
            if "evaluation_strategy" not in training_parameters:
                original_training_arguments = upstream.TrainingArguments

                def compatible_training_arguments(*args, **kwargs):
                    kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
                    return original_training_arguments(*args, **kwargs)

                upstream.TrainingArguments = compatible_training_arguments
            trainer_parameters = inspect.signature(upstream.Trainer).parameters
            if "tokenizer" not in trainer_parameters:
                original_trainer = upstream.Trainer

                def compatible_trainer(*args, **kwargs):
                    if "tokenizer" in kwargs:
                        kwargs["processing_class"] = kwargs.pop("tokenizer")
                    return original_trainer(*args, **kwargs)

                upstream.Trainer = compatible_trainer
            upstream._styleslip_compat_patched = True
        set_seed(self.seed)
        train = pd.DataFrame({"text": list(train_texts), "label": train_labels.tolist()})
        validation = pd.DataFrame(
            {"text": list(validation_texts), "label": validation_labels.tolist()}
        )
        upstream.fine_tune(
            train,
            validation,
            str(self.checkpoints),
            {0: "human", 1: "machine"},
            {"human": 0, "machine": 1},
            self.model_name,
        )
        return self

    def _load_for_inference(self):
        if self._model is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Missing Transformer baseline dependencies") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.best_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.best_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device).eval()
        return self._tokenizer, self._model

    def predict_score(
        self, texts: Sequence[str], *, show_progress: bool = False, split: str = "test"
    ) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        tokenizer, model = self._load_for_inference()

        def collate(batch):
            return tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )

        loader = DataLoader(list(texts), batch_size=16, collate_fn=collate, num_workers=0)
        scores: list[np.ndarray] = []
        iterator = tqdm(loader, desc=f"XLM-R {split}", disable=not show_progress)
        with torch.inference_mode():
            for batch in iterator:
                logits = model(
                    **{key: value.to(model.device) for key, value in batch.items()}
                ).logits
                scores.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def save_artifact(self, _run_dir: Path) -> str:
        if not self.best_model.is_dir():
            raise FileNotFoundError("the official trainer did not save its best model")
        return str(self.best_model.relative_to(self.run_dir))

    def specification(self) -> dict[str, object]:
        return {
            "family": "XLM-RoBERTa sequence classification",
            "model": self.model_name,
            "implementation": str(SEMEVAL_SCRIPT.relative_to(PROJECT_ROOT)),
            "upstream_commit": SEMEVAL_COMMIT,
            "upstream_settings": {
                "learning_rate": 2e-5,
                "train_batch_size": 16,
                "eval_batch_size": 16,
                "epochs": 3,
                "weight_decay": 0.01,
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "tokenizer_truncation": True,
            },
            "adapter_changes": [
                "uses each benchmark's existing validation split instead of re-splitting train 80/20",
                "adds machine-class probability export for common AUROC/AUPRC/Brier evaluation",
                "maps renamed Transformers 5 API arguments without changing training values when needed",
            ],
            "seed": self.seed,
        }


class OfficialFastDetectGPTAdapter:
    """Instantiate and call the released ``scripts/local_infer.py`` detector."""

    requires_training = False
    native_threshold = 0.5

    def __init__(
        self,
        *,
        sampling_model: str,
        scoring_model: str,
        device: str,
        cache_dir: Path,
    ) -> None:
        if (sampling_model, scoring_model) not in FAST_DETECT_PAIRS:
            raise ValueError(
                "the official local detector has no calibrated probability for "
                f"{sampling_model}/{scoring_model}"
            )
        self.sampling_model_name = sampling_model
        self.scoring_model_name = scoring_model
        self.device = device
        self.cache_dir = cache_dir
        self.detector = None

    def fit(self, *_args, **_kwargs) -> "OfficialFastDetectGPTAdapter":
        return self

    def _load(self) -> None:
        if self.detector is not None:
            return
        if self.device.startswith("cuda"):
            required = (
                8.0
                if self.sampling_model_name == self.scoring_model_name == "gpt-neo-2.7B"
                else 18.5
            )
            _require_cuda_capacity(required, "Fast-DetectGPT")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(FAST_DETECT_SCRIPTS))
        try:
            upstream = _load_module(
                "styleslip_upstream_fast_detect_local_infer",
                FAST_DETECT_SCRIPTS / "local_infer.py",
            )
            self.detector = upstream.FastDetectGPT(
                Namespace(
                    sampling_model_name=self.sampling_model_name,
                    scoring_model_name=self.scoring_model_name,
                    device=self.device,
                    cache_dir=str(self.cache_dir),
                )
            )
        finally:
            if sys.path[0] == str(FAST_DETECT_SCRIPTS):
                sys.path.pop(0)

    def predict_score(
        self, texts: Sequence[str], *, show_progress: bool = False, split: str = "test"
    ) -> np.ndarray:
        from tqdm.auto import tqdm

        self._load()
        iterator = tqdm(texts, desc=f"Fast-DetectGPT {split}", disable=not show_progress)
        return np.asarray(
            [float(self.detector.compute_prob(text)[0]) for text in iterator],
            dtype=np.float64,
        )

    def save_artifact(self, run_dir: Path) -> str:
        from .common import write_json

        write_json(run_dir / "detector_config.json", self.specification())
        return "detector_config.json"

    def specification(self) -> dict[str, object]:
        return {
            "family": "Fast-DetectGPT",
            "sampling_model": self.sampling_model_name,
            "scoring_model": self.scoring_model_name,
            "criterion": "official get_sampling_discrepancy_analytic",
            "probability_calibration": "official local_infer.py Gaussian calibration",
            "implementation": str(
                (FAST_DETECT_SCRIPTS / "local_infer.py").relative_to(PROJECT_ROOT)
            ),
            "upstream_commit": FAST_DETECT_COMMIT,
            "tokenization": "official truncation=True with model tokenizer limit",
            "configuration_role": (
                "paper black-box configuration"
                if (self.sampling_model_name, self.scoring_model_name)
                == ("gpt-j-6B", "gpt-neo-2.7B")
                else "official supported detector configuration"
            ),
        }


class OfficialBinocularsAdapter:
    """Call the released Binoculars package without changing model or precision."""

    requires_training = False

    def __init__(
        self,
        *,
        observer_model: str,
        performer_model: str,
        max_length: int,
        mode: str,
        batch_size: int,
    ) -> None:
        self.observer_model = observer_model
        self.performer_model = performer_model
        self.max_length = max_length
        self.mode = mode
        self.batch_size = batch_size
        self.detector = None

    @property
    def native_threshold(self) -> float:
        thresholds = {"accuracy": 0.9015310749276843, "low-fpr": 0.8536432310785527}
        return 1.0 - thresholds[self.mode]

    def fit(self, *_args, **_kwargs) -> "OfficialBinocularsAdapter":
        return self

    def _load(self) -> None:
        if self.detector is not None:
            return
        _require_binoculars_capacity()
        package = _load_module(
            "styleslip_upstream_binoculars",
            BINOCULARS_ROOT / "binoculars" / "__init__.py",
            package_dir=BINOCULARS_ROOT / "binoculars",
        )
        self.detector = package.Binoculars(
            observer_name_or_path=self.observer_model,
            performer_name_or_path=self.performer_model,
            use_bfloat16=True,
            max_token_observed=self.max_length,
            mode=self.mode,
        )

    def predict_score(
        self, texts: Sequence[str], *, show_progress: bool = False, split: str = "test"
    ) -> np.ndarray:
        from tqdm.auto import tqdm

        self._load()
        values: list[float] = []
        starts = range(0, len(texts), self.batch_size)
        iterator = tqdm(starts, desc=f"Binoculars {split}", disable=not show_progress)
        for start in iterator:
            batch = list(texts[start : start + self.batch_size])
            ratios = self.detector.compute_score(batch)
            values.extend(1.0 - float(value) for value in ratios)
        return np.asarray(values, dtype=np.float64)

    def save_artifact(self, run_dir: Path) -> str:
        from .common import write_json

        write_json(run_dir / "detector_config.json", self.specification())
        return "detector_config.json"

    def specification(self) -> dict[str, object]:
        return {
            "family": "Binoculars",
            "observer_model": self.observer_model,
            "performer_model": self.performer_model,
            "dtype": "bfloat16",
            "max_token_observed": self.max_length,
            "mode": self.mode,
            "native_ratio_threshold": 1.0 - self.native_threshold,
            "implementation": str(
                (BINOCULARS_ROOT / "binoculars" / "detector.py").relative_to(PROJECT_ROOT)
            ),
            "upstream_commit": BINOCULARS_COMMIT,
            "score_adapter": "1 - official ratio, so larger means more AI-like",
        }


__all__ = [
    "BINOCULARS_COMMIT",
    "FAST_DETECT_COMMIT",
    "OfficialBinocularsAdapter",
    "OfficialFastDetectGPTAdapter",
    "SEMEVAL_COMMIT",
    "SemEvalXLMRAdapter",
]
