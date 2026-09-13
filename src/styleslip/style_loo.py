"""StyleDistance leave-one-out features over complete documents.

For every alphabetic token, the token is removed from its local sentence
context and that context is encoded again. Three fixed-length channels capture
the deletion magnitude, its alignment with the document mean direction, and
the drift between consecutive deletion directions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import nltk
import numpy as np
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer
from numpy.typing import NDArray

from .wave import fixed_length_wave


MODEL_NAME = "StyleDistance/styledistance_synthetic_only"
STYLE_CHANNEL_NAMES = (
    "loo_cosine_distance_half",
    "loo_direction_alignment",
    "loo_direction_drift_half",
)


class StyleEncoder(Protocol):
    """The subset of the SentenceTransformer API used by StyleSlip."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> NDArray[np.floating[Any]]: ...


@dataclass(frozen=True, slots=True)
class StyleLooConfig:
    """Serializable extraction and wave-construction settings."""

    output_length: int = 256
    sigma: float = 2.0
    gaussian_truncate: float = 4.0
    context_tokens: int = 128
    max_tokens: int | None = None
    encode_batch_size: int = 256
    embedding_output_chunk_size: int = 2048

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("output_length", self.output_length, 2),
            ("context_tokens", self.context_tokens, 2),
            ("encode_batch_size", self.encode_batch_size, 1),
            ("embedding_output_chunk_size", self.embedding_output_chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
                raise TypeError("max_tokens must be an integer or None")
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive when supplied")
        for name, value in (
            ("sigma", self.sigma),
            ("gaussian_truncate", self.gaussian_truncate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_length": self.output_length,
            "sigma": float(self.sigma),
            "gaussian_truncate": float(self.gaussian_truncate),
            "context_tokens": self.context_tokens,
            "max_tokens": self.max_tokens,
            "encode_batch_size": self.encode_batch_size,
            "embedding_output_chunk_size": self.embedding_output_chunk_size,
        }


@dataclass(frozen=True, slots=True)
class TokenContext:
    tokens: tuple[str, ...]
    positions: tuple[int, ...]


@dataclass(slots=True)
class StyleWaveBatch:
    waves: NDArray[np.float32]
    token_counts: NDArray[np.int32]
    context_counts: NDArray[np.int32]
    eligible_token_counts: NDArray[np.int32]
    document_direction_concentrations: NDArray[np.float32]
    magnitude_sum: float

    @property
    def eligible_token_count(self) -> int:
        return int(self.eligible_token_counts.sum())

    @property
    def mean_leave_one_out_cosine_distance_half(self) -> float:
        count = self.eligible_token_count
        return self.magnitude_sum / count if count else float("nan")


def tokenize_document_contexts(
    text: str,
    *,
    context_tokens: int = 128,
    max_tokens: int | None = None,
    sentence_tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> tuple[TokenContext, ...]:
    """Tokenize a complete document into bounded local contexts."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise TypeError("context_tokens must be an integer")
    if context_tokens < 2:
        raise ValueError("context_tokens must be at least 2")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer or None")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive when supplied")

    split = sentence_tokenizer or (lambda value: nltk.sent_tokenize(value, language="english"))
    sentences = list(split(text))
    if not sentences and text.strip():
        sentences = [text]

    tokenizer = TreebankWordTokenizer()
    contexts: list[TokenContext] = []
    next_position = 0
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if max_tokens is not None:
            remaining = max_tokens - next_position
            if remaining <= 0:
                break
            tokens = tokens[:remaining]
        for start in range(0, len(tokens), context_tokens):
            chunk = tuple(tokens[start : start + context_tokens])
            if chunk:
                begin = next_position + start
                contexts.append(
                    TokenContext(chunk, tuple(range(begin, begin + len(chunk))))
                )
        next_position += len(tokens)
        if max_tokens is not None and next_position >= max_tokens:
            break
    return tuple(contexts)


def _normalized_embeddings(
    encoder: StyleEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> NDArray[np.float32]:
    if not texts:
        raise ValueError("cannot encode an empty text collection")
    raw = encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] != len(texts):
        raise ValueError(
            f"style encoder returned shape {values.shape}; expected ({len(texts)}, D)"
        )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("style encoder returned a zero or non-finite embedding")
    return (values / norms).astype(np.float32, copy=False)


def build_style_loo_waves(
    encoder: StyleEncoder,
    texts: Sequence[str],
    *,
    config: StyleLooConfig | None = None,
    sentence_tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> StyleWaveBatch:
    """Build three StyleDistance LOO channels for each supplied document."""

    if not texts:
        raise ValueError("texts must contain at least one document")
    cfg = config or StyleLooConfig()
    if not isinstance(cfg, StyleLooConfig):
        raise TypeError("config must be a StyleLooConfig or None")

    contexts_by_document = [
        tokenize_document_contexts(
            text,
            context_tokens=cfg.context_tokens,
            max_tokens=cfg.max_tokens,
            sentence_tokenizer=sentence_tokenizer,
        )
        for text in texts
    ]
    token_counts = np.asarray(
        [sum(len(context.tokens) for context in contexts) for contexts in contexts_by_document],
        dtype=np.int32,
    )
    if np.any(token_counts == 0):
        raise ValueError(
            f"document(s) produced no tokens: {np.flatnonzero(token_counts == 0).tolist()}"
        )

    detokenizer = TreebankWordDetokenizer()
    full_texts: list[str] = []
    variant_texts: list[str] = []
    mappings: list[tuple[int, int, int, int]] = []
    for document_index, contexts in enumerate(contexts_by_document):
        for context in contexts:
            tokens = list(context.tokens)
            if len(tokens) < 2:
                continue
            full_index = len(full_texts)
            full_texts.append(detokenizer.detokenize(tokens))
            for local_index, (token, position) in enumerate(zip(tokens, context.positions)):
                if not any(character.isalpha() for character in token):
                    continue
                variant_index = len(variant_texts)
                variant = tokens[:local_index] + tokens[local_index + 1 :]
                variant_texts.append(detokenizer.detokenize(variant))
                mappings.append((document_index, position, full_index, variant_index))
    if not mappings:
        raise ValueError("documents produced no eligible alphabetic leave-one-out tokens")

    full_embeddings = _normalized_embeddings(
        encoder, full_texts, batch_size=cfg.encode_batch_size
    )
    dimension = int(full_embeddings.shape[1])
    magnitudes = [np.full(int(count), np.nan, dtype=np.float32) for count in token_counts]
    directions = [
        np.full((int(count), dimension), np.nan, dtype=np.float32)
        for count in token_counts
    ]
    for start in range(0, len(variant_texts), cfg.embedding_output_chunk_size):
        stop = min(start + cfg.embedding_output_chunk_size, len(variant_texts))
        variant_embeddings = _normalized_embeddings(
            encoder, variant_texts[start:stop], batch_size=cfg.encode_batch_size
        )
        if variant_embeddings.shape[1] != dimension:
            raise ValueError("original and deletion embedding dimensions disagree")
        for document_index, position, full_index, variant_index in mappings[start:stop]:
            full = full_embeddings[full_index]
            removed = variant_embeddings[variant_index - start]
            cosine = float(np.clip(np.dot(full, removed), -1.0, 1.0))
            magnitudes[document_index][position] = np.clip((1.0 - cosine) / 2.0, 0.0, 1.0)
            difference = full - removed
            norm = float(np.linalg.norm(difference))
            if norm > 1e-8:
                directions[document_index][position] = difference / norm

    waves: list[NDArray[np.float32]] = []
    concentrations: list[float] = []
    eligible_counts: list[int] = []
    magnitude_sum = 0.0
    for document_index, token_count in enumerate(token_counts):
        valid = np.isfinite(directions[document_index]).all(axis=1)
        if not np.any(valid):
            raise ValueError(
                f"document {document_index} produced no finite leave-one-out directions"
            )
        valid_positions = np.flatnonzero(valid)
        unit_directions = directions[document_index][valid]
        mean_direction = unit_directions.mean(axis=0)
        concentration = float(np.linalg.norm(mean_direction))
        concentrations.append(concentration)
        if concentration > 1e-8:
            mean_direction /= concentration

        alignment = np.full(int(token_count), np.nan, dtype=np.float32)
        alignment[valid] = np.clip(
            (unit_directions @ mean_direction + 1.0) / 2.0, 0.0, 1.0
        )
        drift = np.full(int(token_count), np.nan, dtype=np.float32)
        drift[valid_positions[0]] = 0.0
        if len(valid_positions) > 1:
            adjacent_cosines = np.sum(unit_directions[1:] * unit_directions[:-1], axis=1)
            drift[valid_positions[1:]] = np.clip(
                (1.0 - adjacent_cosines) / 2.0, 0.0, 1.0
            )

        eligible_counts.append(int(len(valid_positions)))
        magnitude_sum += float(magnitudes[document_index][valid].sum(dtype=np.float64))
        channels = [
            fixed_length_wave(
                values,
                output_length=cfg.output_length,
                sigma=cfg.sigma,
                gaussian_truncate=cfg.gaussian_truncate,
            )
            for values in (magnitudes[document_index], alignment, drift)
        ]
        waves.append(np.stack(channels, axis=0).astype(np.float32, copy=False))

    return StyleWaveBatch(
        waves=np.stack(waves, axis=0).astype(np.float32, copy=False),
        token_counts=token_counts,
        context_counts=np.asarray([len(value) for value in contexts_by_document], dtype=np.int32),
        eligible_token_counts=np.asarray(eligible_counts, dtype=np.int32),
        document_direction_concentrations=np.asarray(concentrations, dtype=np.float32),
        magnitude_sum=magnitude_sum,
    )


__all__ = [
    "MODEL_NAME",
    "STYLE_CHANNEL_NAMES",
    "StyleEncoder",
    "StyleLooConfig",
    "StyleWaveBatch",
    "TokenContext",
    "build_style_loo_waves",
    "tokenize_document_contexts",
]

