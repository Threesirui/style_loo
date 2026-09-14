"""Tests for paper/source-traceable classical baselines."""

from __future__ import annotations

import json

import numpy as np

from styleslip.baselines.runner import main
from styleslip.baselines.stylometric_lr import (
    BERTAA_CODE_FEATURE_NAMES,
    EXTRACTED_FEATURE_NAMES,
    PUNCTUATION,
    StylometricLR,
    extract_stylometric_features,
)
from styleslip.baselines.tfidf_svm import TfidfSVM


def test_bertaa_feature_values_and_source_code_selection() -> None:
    values = extract_stylometric_features("Aa 1!")
    lookup = dict(zip(EXTRACTED_FEATURE_NAMES, values))
    assert len(EXTRACTED_FEATURE_NAMES) == 55
    assert len(BERTAA_CODE_FEATURE_NAMES) == 53
    assert PUNCTUATION == ("!", "-", ":", "?", ".", ",", ";", "'", "/", "(", ")", "&")
    assert lookup["avg_word_length"] == 2.0
    assert lookup["text_length"] == 5.0
    assert lookup["word_count"] == 2.0
    assert lookup["short_word_count"] == 2.0
    assert lookup["digit_ratio"] == 0.2
    assert lookup["capital_ratio"] == 0.2
    assert lookup["letter_a_ratio"] == 0.4
    assert lookup["digit_1_ratio"] == 0.2
    assert lookup["punct_21_ratio"] == 0.2
    assert lookup["type_token_ratio"] == 1.0
    assert "text_length" not in StylometricLR("bertaa_code").feature_names
    assert "text_length" in StylometricLR("paper_full").feature_names
    unicode_lookup = dict(zip(EXTRACTED_FEATURE_NAMES, extract_stylometric_features("ß word")))
    assert unicode_lookup["letter_s_ratio"] == 0.0


def test_pan25_vectorizer_matches_official_constructor() -> None:
    model = TfidfSVM(profile="pan25")
    params = model.vectorizer.get_params()
    assert params["analyzer"] == "word"
    assert params["ngram_range"] == (1, 4)
    assert params["max_features"] == 1000
    assert params["sublinear_tf"] is False
    assert model.classifier.get_params()["C"] == 1.0


def test_enhanced_vocabulary_is_fit_on_train_only() -> None:
    train = ["human prose varies naturally", "machine prose repeats regularly"]
    validation = ["xylophone human", "xylophone machine"]
    model = TfidfSVM(
        profile="enhanced", word_max_features=100, char_max_features=100
    ).fit(
        train,
        np.asarray([0, 1]),
        validation_texts=validation,
        validation_labels=np.asarray([0, 1]),
        c_grid=(0.1, 1.0),
    )
    transformers = dict(model.vectorizer.transformer_list)
    assert "xylophone" not in transformers["word"].vocabulary_
    assert set(model.validation_scores_) == {"0.1", "1"}
    assert np.all((model.predict_score(validation) >= 0) & (model.predict_score(validation) <= 1))


def _write_m4(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (text, label) in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "id": f"{path.stem}-{index}",
                        "text": text,
                        "label": label,
                        "model": "human" if label == 0 else "toy-generator",
                        "source": "toy-domain",
                    }
                )
                + "\n"
            )


def test_baseline_runner_smoke(tmp_path) -> None:
    base = tmp_path / "data" / "SemEval2024-M4" / "SubtaskA"
    train_rows = [
        ("A human writer varies words and rhythm.", 0),
        ("Personal memories make this human passage distinct.", 0),
        ("Generated response follows a regular template.", 1),
        ("Machine output follows another regular template.", 1),
    ]
    validation_rows = [
        ("Human prose includes a surprising aside.", 0),
        ("Generated prose uses a predictable response.", 1),
    ]
    test_rows = [
        ("A person recalls an unusual afternoon.", 0),
        ("The generated answer is orderly and predictable.", 1),
    ]
    _write_m4(base / "subtaskA_train_monolingual.jsonl", train_rows)
    with (base / "subtaskA_train_monolingual.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "blank-record",
                    "text": "\r",
                    "label": 1,
                    "model": "davinci",
                    "source": "chinese",
                }
            )
            + "\n{broken json\n"
        )
    _write_m4(base / "subtaskA_dev_monolingual.jsonl", validation_rows)
    _write_m4(base / "subtaskA_test_monolingual.jsonl", test_rows)
    output = tmp_path / "outputs"
    status = main(
        [
            "--dataset",
            "m4",
            "--scenario",
            "monolingual",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
            "--samples-per-class",
            "2",
            "--method",
            "enhanced_tfidf_svm",
            "--method",
            "stylometric_lr",
            "--c-grid",
            "1",
        ]
    )
    assert status == 0
    manifests = list(output.rglob("experiment_manifest.json"))
    assert len(manifests) == 2
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["fit_scope"]["test_or_ood_used_for_fit"] is False
        invalid_train = manifest["split_details"]["invalid_records"]["train"]
        assert invalid_train["total"] == 2
        assert invalid_train["reasons"] == {"invalid JSON": 1, "text is empty": 1}
        assert (path.parent / "model.pkl").is_file()
        assert (path.parent / "test_predictions.csv").is_file()
