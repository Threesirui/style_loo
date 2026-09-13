# Classical baselines with source provenance

This directory provides a zero-install launcher for two requested baselines. The implementation lives in
`src/styleslip/baselines` so it is also available from the installed package.

## What is reproduced

| Method | Default implementation | Reproduction status |
| --- | --- | --- |
| `stylometric_lr` | 53 BertAA source-code features + Logistic Regression | Behavioral reimplementation of the public notebook linked by the BertAA paper; also the feature recipe cited by Team Innovative |
| `enhanced_tfidf_svm` | word 1–2 + character 3–5 TF-IDF, LinearSVC, validation-selected C | Declared extension of the official PAN'25 TF-IDF-SVM code |
| `pan25_tfidf_svm` | word 1–4 TF-IDF, 1,000 features, default LinearSVC | Configuration-compatible reproduction of the official PAN'25 source |

The word/character combination called **Enhanced TF-IDF-SVM** is not claimed to be an unchanged model from
one paper. It is an auditable extension of the official PAN implementation. Use `pan25_tfidf_svm` when an exact
upstream configuration is required.

The public BertAA notebook computes 55 features, but its classifier drops `text_length` and `word_count` and uses
53. Therefore `bertaa_code` is the default. `--stylo-profile paper_full` includes all 55 values described by the
paper and is reported as a specification variant, not as exact source-code behavior. A second discrepancy is that
the papers call the final feature “hapax legomena”, while the source actually computes a case-sensitive type-token
ratio; both profiles retain the executable source behavior and record the feature as `type_token_ratio`.

ACL's supplementary ZIP for Team Innovative was also inspected: it contains only five LaTeX/bibliography files
and no executable model code or software license. Consequently, a strict reproduction of that team's unpublished
implementation cannot be claimed. The default instead follows the downloadable BertAA notebook that defines the
feature recipe cited and repeated by Team Innovative.

## Sources

- PAN'25 official repository, commit `b12f844ecce7da77fee2fdc23f27588838e14425`, Apache-2.0:
  <https://github.com/pan-webis-de/pan25-generative-ai-authorship-verification>
- BertAA paper and its public Colab notebook:
  <https://aclanthology.org/2020.icon-main.16/> and
  <https://colab.research.google.com/drive/1m4anWkkb8tz3fKvzJFytygBkqCTdZ8bo>
- Team Innovative at SemEval-2024 Task 8:
  <https://aclanthology.org/2024.semeval-1.171/>

The BertAA notebook does not state a software license. Its code is not vendored; this project independently
implements the published behavior and records the downloaded notebook SHA-256 in each experiment manifest.

## Run

```powershell
$python = "C:\Users\three\.conda\envs\TextWave\python.exe"

# Requested two baselines (default methods)
& $python .\baselines\run.py --dataset m4 --scenario monolingual

# Exact PAN'25 source configuration
& $python .\baselines\run.py --dataset m4 --scenario monolingual `
  --method pan25_tfidf_svm

# Exact BertAA notebook feature selection
& $python .\baselines\run.py --dataset deepfake --scenario unseen_models `
  --method stylometric_lr --stylo-profile bertaa_code

# Inspect all planned inputs without loading text
& $python .\baselines\run.py --dataset raid --scenario clean --dry-run
```

The dataset discovery, label normalization, deterministic sampling, RAID `source_id` group split, and blind-test
handling are shared with StyleSlip. Vectorizers and classifiers see only `train`; C and the optional F1 threshold
see only `validation`; test and OOD data are evaluation-only. Both validation-selected and fixed-0.5 metrics are
written so results can be compared under the StyleSlip protocol or the upstream classifier decision boundary.

Outputs are stored under `outputs/baselines/<method>/<dataset>/<scenario>/<case>/<hash>/` and include the model,
manifest, metrics, and per-split predictions. The manifest records upstream provenance, runtime versions, split
sizes, overlap counts, and exactly which splits affected fitting.
