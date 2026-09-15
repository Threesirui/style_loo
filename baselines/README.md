# Baselines with source provenance

This directory provides a zero-install launcher for the requested baselines. The implementation lives in
`src/styleslip/baselines` so it is also available from the installed package.

## What is reproduced

| Method | Default implementation | Reproduction status |
| --- | --- | --- |
| `stylometric_lr` | 53 BertAA source-code features + Logistic Regression | Behavioral reimplementation of the public notebook linked by the BertAA paper; also the feature recipe cited by Team Innovative |
| `enhanced_tfidf_svm` | word 1–2 + character 3–5 TF-IDF, LinearSVC, validation-selected C | Declared extension of the official PAN'25 TF-IDF-SVM code |
| `pan25_tfidf_svm` | word 1–4 TF-IDF, 1,000 features, default LinearSVC | Configuration-compatible reproduction of the official PAN'25 source |
| `xlm_roberta_base_ft` | XLM-RoBERTa-base, 3 epochs, LR 2e-5, batch 16, weight decay 0.01 | Calls the official SemEval-2024 Task 8 `fine_tune` function |
| `fast_detect_gpt` | GPT-J-6B sampler + GPT-Neo-2.7B scorer | Calls the official Fast-DetectGPT local detector in the paper's black-box setting |
| `binoculars` | Falcon-7B observer + Falcon-7B-Instruct performer, bfloat16, 512 tokens | Calls the official Binoculars package and published global threshold |

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
- SemEval-2024 Task 8 official baseline, commit `d8350c840bc505eaba06b4baf69993c2d18fef5e`, Apache-2.0:
  <https://github.com/mbzuai-nlp/SemEval2024-task8>
- Fast-DetectGPT official repository, commit `971b05202bac2bb504d60c0ac0812fea7a8f7c82`, MIT:
  <https://github.com/baoguangsheng/fast-detect-gpt>
- Binoculars official repository, commit `c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8`, BSD-3-Clause:
  <https://github.com/ahans30/Binoculars>

The BertAA notebook does not state a software license. Its code is not vendored; this project independently
implements the published behavior and records the downloaded notebook SHA-256 in each experiment manifest.

The three neural/zero-shot baselines are different: their released source files are fetched verbatim into the
ignored `baselines/upstream/` working directory and imported at runtime. `upstream_manifest.json` pins commits,
licenses, entry points, and SHA-256 values. Recreate or verify that mirror with:

```powershell
& .\baselines\fetch_upstream.ps1
```

The adapter never reimplements the detector equations. It only maps benchmark records to the official API,
exports a common machine-positive score, and records unavoidable protocol changes. In particular, XLM-R uses
each benchmark's existing validation split instead of the SemEval script's internal 80/20 split. Binoculars'
ratio is exported as `1 - ratio`, so its published decision rule remains exactly equivalent after reversing the
threshold direction.

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

# XLM-RoBERTa-base fine-tuning on all three benchmark families
& $python .\baselines\run.py --dataset all --method xlm_roberta_base_ft

# Fast-DetectGPT paper black-box setting (GPT-J-6B / GPT-Neo-2.7B)
& $python .\baselines\run.py --dataset all --method fast_detect_gpt

# Binoculars paper/repository setting and its low-FPR global threshold
& $python .\baselines\run.py --dataset all --method binoculars --threshold-mode native
```

Install adapter dependencies with `pip install -e ".[neural-baselines]"`. Model weights are downloaded by the
original Hugging Face calls on first execution and cached under `model/` (Fast-DetectGPT) or the normal Hugging
Face cache (XLM-R and Binoculars).

### Hardware fidelity

The Fast-DetectGPT paper reports a Tesla A100 80 GB; its GPT-J-6B plus GPT-Neo-2.7B black-box pair needs more
than a 16 GB card in the released fp16 loader. The repository also exposes a lower-memory single-model demo:

```powershell
& $python .\baselines\run.py --dataset all --method fast_detect_gpt `
  --fast-sampling-model gpt-neo-2.7B --fast-scoring-model gpt-neo-2.7B
```

That command is an official supported configuration, but it is labelled as a variant rather than the paper's
main black-box setting. Official Binoculars loads two 7B Falcon models in bfloat16 and normally needs two GPUs
with roughly 15 GB each, or one GPU around 30 GB. The runner fails before downloading weights when visible GPU
capacity cannot fit; it never silently quantizes or substitutes smaller models.

The dataset discovery, label normalization, deterministic sampling, RAID `source_id` group split, and blind-test
handling are shared with StyleSlip. Vectorizers and trained classifiers see only `train`; C, XLM-R checkpoint
selection, and the optional F1 threshold see only `validation`; test and OOD data are evaluation-only.
Validation-selected, fixed-0.5, and native-threshold metrics are written so results can be compared under the
common StyleSlip protocol and the upstream decision boundary. Brier score is `null` for unbounded Binoculars
scores because Brier is defined for probabilities.

By default, `--samples-per-class` is unlimited: every valid record in each split is used. Pass an integer to run a
bounded experiment, for example `--samples-per-class 1000`. Full Enhanced TF-IDF can require substantial memory
because its word and character vocabularies are fitted on the complete training corpus.

Progress output is enabled by default and reports split loading/counts, feature construction, each SVM `C`
candidate, threshold selection, and validation/test/OOD metric summaries. Use `--no-progress` for quiet batch logs.

Baseline runs default to `--invalid-record-policy skip`. A malformed individual record (for example invalid JSON,
whitespace-only text, an illegal label, or incomplete required fields) is skipped and audited under
`split_details.invalid_records` in the manifest. The report contains totals, reasons, and at most 20 source
locations. Structural errors such as a missing file or an invalid CSV header still stop the run. Use
`--invalid-record-policy error` for the original fail-fast behavior. Source dataset files are never modified.

Outputs are stored under `outputs/baselines/<method>/<dataset>/<scenario>/<case>/<hash>/` and include the model,
manifest, metrics, and per-split predictions. The manifest records upstream provenance, runtime versions, split
sizes, overlap counts, and exactly which splits affected fitting.
