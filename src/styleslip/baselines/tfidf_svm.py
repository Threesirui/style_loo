"""PAN'25-derived TF-IDF + linear SVM baselines."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


PAN25_COMMIT = "b12f844ecce7da77fee2fdc23f27588838e14425"


def _f1(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = scores >= 0.5
    true_positive = int(np.sum((labels == 1) & predictions))
    false_positive = int(np.sum((labels == 0) & predictions))
    false_negative = int(np.sum((labels == 1) & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


class TfidfSVM:
    """LinearSVC over either the exact PAN'25 or an explicitly enhanced TF-IDF.

    ``pan25`` reproduces the task repository's training code: word 1--4 grams,
    at most 1,000 features, and a default ``LinearSVC``. ``enhanced`` retains
    that model family while combining word 1--2 and character 3--5 grams.
    """

    def __init__(
        self,
        profile: Literal["pan25", "enhanced"] = "enhanced",
        *,
        c: float = 1.0,
        word_max_features: int | None = 100_000,
        char_max_features: int | None = 200_000,
    ) -> None:
        if profile not in {"pan25", "enhanced"}:
            raise ValueError("profile must be pan25 or enhanced")
        if c <= 0:
            raise ValueError("C must be positive")
        self.profile = profile
        self.c = float(c)
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.vectorizer = self._new_vectorizer()
        self.classifier = LinearSVC() if profile == "pan25" and c == 1.0 else LinearSVC(C=self.c)
        self.validation_scores_: dict[str, float] = {}

    def _new_vectorizer(self):
        if self.profile == "pan25":
            # Exact constructor arguments in the official PAN'25 repository.
            return TfidfVectorizer(ngram_range=(1, 4), max_features=1000)
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=self.word_max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        max_features=self.char_max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    @staticmethod
    def _scores(classifier: LinearSVC, matrix) -> np.ndarray:
        # PAN calls LinearSVC._predict_proba_lr.  Its binary positive-class
        # output is sigmoid(decision_function); keep a public-API fallback.
        if hasattr(classifier, "_predict_proba_lr"):
            return np.asarray(classifier._predict_proba_lr(matrix)[:, 1], dtype=np.float64)
        decision = np.asarray(classifier.decision_function(matrix), dtype=np.float64)
        decision = np.clip(decision, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-decision))

    def fit(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        *,
        validation_texts: Sequence[str] | None = None,
        validation_labels: np.ndarray | None = None,
        c_grid: Sequence[float] | None = None,
        show_progress: bool = False,
    ) -> "TfidfSVM":
        labels = np.asarray(labels, dtype=np.int64)
        if show_progress:
            print(
                f"Fitting {self.profile} TF-IDF vectorizer on {len(texts)} training documents",
                flush=True,
            )
        train_matrix = self.vectorizer.fit_transform(texts)
        if show_progress:
            print(
                f"Training sparse matrix: {train_matrix.shape[0]} × {train_matrix.shape[1]}",
                flush=True,
            )
        candidates = [self.c]
        if self.profile == "enhanced" and c_grid:
            candidates = sorted({float(value) for value in c_grid})
            if not candidates or any(value <= 0 for value in candidates):
                raise ValueError("all C candidates must be positive")
        if len(candidates) > 1 and (validation_texts is None or validation_labels is None):
            raise ValueError("validation data are required to tune C")

        validation_matrix = (
            self.vectorizer.transform(validation_texts) if validation_texts is not None else None
        )
        fitted: list[tuple[float, float, LinearSVC]] = []
        for value in candidates:
            if show_progress:
                print(f"Training LinearSVC with C={value:g}", flush=True)
            classifier = clone(self.classifier).set_params(C=value)
            classifier.fit(train_matrix, labels)
            score = (
                _f1(np.asarray(validation_labels, dtype=np.int64), self._scores(classifier, validation_matrix))
                if validation_matrix is not None and validation_labels is not None
                else float("nan")
            )
            self.validation_scores_[format(value, "g")] = score
            if show_progress and np.isfinite(score):
                print(f"Validation F1 at C={value:g}: {score:.6f}", flush=True)
            fitted.append((score, value, classifier))
        # Highest validation F1; deterministic tie break favours stronger regularization.
        _, self.c, self.classifier = max(fitted, key=lambda item: (item[0], -item[1]))
        if show_progress:
            print(f"Selected LinearSVC C={self.c:g}", flush=True)
        return self

    def predict_score(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
        split: str = "evaluation",
    ) -> np.ndarray:
        if show_progress:
            print(f"Transforming and scoring {len(texts)} {split} documents", flush=True)
        return self._scores(self.classifier, self.vectorizer.transform(texts))

    def specification(self) -> dict[str, object]:
        if self.profile == "pan25":
            features: dict[str, object] = {
                "analyzer": "word",
                "ngram_range": [1, 4],
                "max_features": 1000,
                "sublinear_tf": False,
                "learned_features": len(getattr(self.vectorizer, "vocabulary_", {})),
            }
        else:
            fitted = dict(self.vectorizer.transformer_list)
            features = {
                "word": {
                    "ngram_range": [1, 2],
                    "max_features": self.word_max_features,
                    "sublinear_tf": True,
                    "learned_features": len(getattr(fitted["word"], "vocabulary_", {})),
                },
                "char": {
                    "ngram_range": [3, 5],
                    "max_features": self.char_max_features,
                    "sublinear_tf": True,
                    "learned_features": len(getattr(fitted["char"], "vocabulary_", {})),
                },
            }
        return {
            "name": "Enhanced TF-IDF-SVM" if self.profile == "enhanced" else "PAN25 TF-IDF-SVM",
            "profile": self.profile,
            "features": features,
            "classifier": {
                "type": "sklearn.svm.LinearSVC",
                "C": self.c,
                "score_conversion": "LinearSVC._predict_proba_lr (sigmoid decision score)",
            },
            "validation_c_scores": self.validation_scores_,
        }


__all__ = ["PAN25_COMMIT", "TfidfSVM"]
