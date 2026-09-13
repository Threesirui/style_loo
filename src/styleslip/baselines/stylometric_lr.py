"""BertAA / Team Innovative stylometric Logistic Regression baseline.

This is a clean Python reimplementation of the feature extraction published in
the public BertAA Colab notebook.  The original notebook is linked from the
BertAA paper and was subsequently used as the feature specification by Team
Innovative at SemEval 2024 Task 8.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression


LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
DIGITS = tuple("0123456789")
# Exact order used by the public BertAA notebook.
PUNCTUATION = ("!", "-", ":", "?", ".", ",", ";", "'", "/", "(", ")", "&")

EXTRACTED_FEATURE_NAMES = (
    "avg_word_length",
    "text_length",
    "word_count",
    "short_word_count",
    "digit_ratio",
    "capital_ratio",
    *(f"letter_{value}_ratio" for value in LETTERS),
    *(f"digit_{value}_ratio" for value in DIGITS),
    *(f"punct_{ord(value):02x}_ratio" for value in PUNCTUATION),
    "type_token_ratio",
)

# The notebook computes 55 columns but selects only these 53 for the LR.
BERTAA_CODE_INDICES = tuple(
    index
    for index, name in enumerate(EXTRACTED_FEATURE_NAMES)
    if name not in {"text_length", "word_count"}
)
BERTAA_CODE_FEATURE_NAMES = tuple(EXTRACTED_FEATURE_NAMES[index] for index in BERTAA_CODE_INDICES)


def extract_stylometric_features(text: str) -> np.ndarray:
    """Return the 55 values computed by the public BertAA source code.

    Tokenization deliberately follows ``str.split`` and lexical richness is the
    notebook's case-sensitive type-token ratio (called "richness" there).
    """

    text = str(text)
    if not text:
        raise ValueError("stylometric features require non-empty text")
    words = text.split()
    if not words:
        raise ValueError("stylometric features require at least one whitespace token")
    text_length = len(text)
    # The notebook applies ``char.lower()`` to one character at a time.  Do not
    # casefold the whole string: e.g. German sharp-s would otherwise become two
    # ASCII ``s`` characters and stop matching the source implementation.
    counts = Counter(char.lower() for char in text)
    values = [
        float(np.mean([len(word) for word in words])),
        float(text_length),
        float(len(words)),
        float(sum(len(word) < 3 for word in words)),
        float(sum(char.isdigit() for char in text) / text_length),
        float(sum(char.isupper() for char in text) / text_length),
    ]
    values.extend(float(counts[value] / text_length) for value in LETTERS)
    values.extend(float(counts[value] / text_length) for value in DIGITS)
    values.extend(float(counts[value] / text_length) for value in PUNCTUATION)
    values.append(float(len(set(words)) / len(words)))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(EXTRACTED_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("failed to extract finite BertAA stylometric features")
    return result


class StylometricLR:
    """Logistic Regression over the BertAA stylometric feature vector.

    ``bertaa_code`` exactly follows the columns selected by the public notebook
    (53 features). ``paper_full`` additionally includes text length and word
    count, which the papers describe but the notebook computes and then omits.
    """

    def __init__(self, profile: Literal["bertaa_code", "paper_full"] = "bertaa_code") -> None:
        if profile not in {"bertaa_code", "paper_full"}:
            raise ValueError("profile must be bertaa_code or paper_full")
        self.profile = profile
        # These are the values reported by both BertAA and Team Innovative.
        self.classifier = LogisticRegression(
            tol=1e-4,
            C=1.0,
            max_iter=100,
            fit_intercept=True,
            random_state=0,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return BERTAA_CODE_FEATURE_NAMES if self.profile == "bertaa_code" else EXTRACTED_FEATURE_NAMES

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        matrix = np.vstack([extract_stylometric_features(text) for text in texts])
        return matrix[:, BERTAA_CODE_INDICES] if self.profile == "bertaa_code" else matrix

    def fit(self, texts: Iterable[str], labels: np.ndarray) -> "StylometricLR":
        self.classifier.fit(self.transform(texts), np.asarray(labels, dtype=np.int64))
        return self

    def predict_score(self, texts: Iterable[str]) -> np.ndarray:
        probabilities = self.classifier.predict_proba(self.transform(texts))
        positive = int(np.flatnonzero(self.classifier.classes_ == 1)[0])
        return np.asarray(probabilities[:, positive], dtype=np.float64)

    def specification(self) -> dict[str, object]:
        return {
            "name": "Stylometric-LR",
            "profile": self.profile,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "classifier": {
                "type": "sklearn.linear_model.LogisticRegression",
                "penalty": "l2",
                "tol": 1e-4,
                "C": 1.0,
                "max_iter": 100,
                "fit_intercept": True,
                "random_state": 0,
                "scaling": False,
            },
        }


__all__ = [
    "BERTAA_CODE_FEATURE_NAMES",
    "EXTRACTED_FEATURE_NAMES",
    "PUNCTUATION",
    "StylometricLR",
    "extract_stylometric_features",
]
