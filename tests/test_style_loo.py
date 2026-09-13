"""Offline regression tests for the extracted Style-LOO algorithm."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from styleslip import (
    STYLE_CHANNEL_NAMES,
    StyleLooConfig,
    build_style_loo_waves,
    tokenize_document_contexts,
)


class FakeStyleEncoder:
    def encode(self, sentences: Sequence[str], **_: object) -> np.ndarray:
        tokenizer = TreebankWordTokenizer()
        rows = []
        for sentence in sentences:
            values = np.zeros(12, dtype=np.float32)
            for index, token in enumerate(tokenizer.tokenize(sentence)):
                code = sum(ord(character) for character in token)
                values[code % 7] += 1.0
                values[7] += len(token)
                values[8] += index + 1
                values[9] += code % 17
                values[10] += sum(character.isalpha() for character in token)
            values[11] = 1.0
            rows.append(values)
        return np.stack(rows)


class RecordingStyleEncoder(FakeStyleEncoder):
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        self.call_sizes.append(len(sentences))
        return super().encode(sentences, **kwargs)


def single_sentence(text: str) -> list[str]:
    return [text]


def test_tokenizer_processes_the_complete_document() -> None:
    text = " ".join(f"word{index}" for index in range(300))
    contexts = tokenize_document_contexts(
        text, context_tokens=64, sentence_tokenizer=single_sentence
    )
    assert sum(len(context.tokens) for context in contexts) == 300
    assert len(contexts) == 5
    assert contexts[-1].positions[-1] == 299


def test_style_loo_has_three_finite_fixed_length_waves() -> None:
    text = " ".join(f"token{index}" for index in range(150))
    result = build_style_loo_waves(
        FakeStyleEncoder(),
        [text],
        config=StyleLooConfig(
            output_length=24,
            sigma=2.0,
            context_tokens=32,
            encode_batch_size=64,
        ),
        sentence_tokenizer=single_sentence,
    )
    assert STYLE_CHANNEL_NAMES == (
        "loo_cosine_distance_half",
        "loo_direction_alignment",
        "loo_direction_drift_half",
    )
    assert result.waves.shape == (1, 3, 24)
    assert np.all(np.isfinite(result.waves))
    assert result.token_counts.tolist() == [150]
    assert result.eligible_token_counts.tolist() == [150]
    assert result.context_counts.tolist() == [5]


def test_optional_prefix_cap_is_explicit() -> None:
    text = " ".join(f"word{index}" for index in range(200))
    contexts = tokenize_document_contexts(
        text,
        context_tokens=32,
        max_tokens=75,
        sentence_tokenizer=single_sentence,
    )
    assert sum(len(context.tokens) for context in contexts) == 75
    assert contexts[-1].positions[-1] == 74


def test_deletion_embedding_output_is_memory_bounded() -> None:
    encoder = RecordingStyleEncoder()
    result = build_style_loo_waves(
        encoder,
        [" ".join(f"token{index}" for index in range(150))],
        config=StyleLooConfig(
            output_length=24,
            context_tokens=32,
            encode_batch_size=64,
            embedding_output_chunk_size=17,
        ),
        sentence_tokenizer=single_sentence,
    )
    assert result.waves.shape == (1, 3, 24)
    assert encoder.call_sizes[0] == 5
    assert max(encoder.call_sizes[1:]) <= 17

