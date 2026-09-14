"""Dataset protocol tests using small local fixtures."""

from __future__ import annotations

import csv
import json

import pytest

from styleslip.datasets import InvalidRecordReport, SourceSpec, discover_cases, load_source


def test_m4_discovery_uses_only_subtask_a(tmp_path) -> None:
    directory = tmp_path / "SemEval2024-M4" / "SubtaskA"
    directory.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        (directory / f"subtaskA_{split}_monolingual.jsonl").write_text("", encoding="utf-8")
    cases = discover_cases(tmp_path, dataset="m4", scenario="monolingual")
    assert len(cases) == 1
    assert cases[0].scenario == "monolingual"
    assert cases[0].protocol == "predefined"
    assert "SubtaskA" in str(cases[0].train.path)


def test_deepfake_labels_are_explicitly_inverted(tmp_path) -> None:
    path = tmp_path / "split.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "domain", "is_human", "model"],
        )
        writer.writeheader()
        writer.writerow({"text": "Human words are here.", "label": 1, "domain": "news", "is_human": "true", "model": ""})
        writer.writerow({"text": "Machine words are here.", "label": 0, "domain": "news", "is_human": "false", "model": "gpt"})
    records = load_source(
        SourceSpec(path, "deepfake", "train"),
        samples_per_class=1,
        seed=42,
    )
    assert sorted((record.model, record.label) for record in records) == [("gpt", 1), ("human", 0)]


def test_raid_derives_labels_and_preserves_source_groups(tmp_path) -> None:
    path = tmp_path / "train.csv"
    fields = ["id", "source_id", "model", "attack", "domain", "generation"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"id": "h", "source_id": "source-h", "model": "human", "attack": "none", "domain": "books", "generation": "Human document words."})
        writer.writerow({"id": "a", "source_id": "source-a", "model": "gpt", "attack": "synonym", "domain": "books", "generation": "Machine document words."})
    records = load_source(SourceSpec(path, "raid", "source"), samples_per_class=1, seed=7)
    assert {record.label for record in records} == {0, 1}
    assert {record.group_id for record in records} == {"source-h", "source-a"}
    assert {record.attack for record in records} == {"none", "synonym"}


def test_tolerant_m4_loader_skips_and_audits_bad_individual_records(tmp_path) -> None:
    path = tmp_path / "split.jsonl"
    rows = [
        {"id": "h", "text": "Valid human text.", "label": 0, "model": "human", "source": "news"},
        {"id": "blank", "text": "\r", "label": 1, "model": "davinci", "source": "chinese"},
        {"id": "a", "text": "Valid generated text.", "label": 1, "model": "gpt", "source": "news"},
    ]
    path.write_text(
        "\n".join([json.dumps(rows[0]), json.dumps(rows[1]), "{broken json", json.dumps(rows[2])]) + "\n",
        encoding="utf-8",
    )
    source = SourceSpec(path, "m4", "train")
    with pytest.raises(ValueError, match="text is empty"):
        load_source(source, samples_per_class=None, seed=42)

    report = InvalidRecordReport()
    records = load_source(
        source,
        samples_per_class=None,
        seed=42,
        invalid_policy="skip",
        invalid_report=report,
    )
    assert {record.id for record in records} == {"h", "a"}
    assert report.total == 2
    assert report.reasons == {"invalid JSON": 1, "text is empty": 1}
    assert report.to_dict()["examples_truncated"] is False
