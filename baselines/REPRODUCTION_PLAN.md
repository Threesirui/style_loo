# Primary neural baseline reproduction plan

## Primary comparison

The primary three-benchmark table uses one predeclared protocol per benchmark family:

| Benchmark | Scenario | Training/evaluation protocol |
| --- | --- | --- |
| M4 | multilingual Subtask A | authored train/dev/test splits; XLM-R is the official multilingual baseline model |
| Deepfake/MAGE | cross domains + cross models | authored train/valid/test; `test_ood_gpt.csv` is evaluation-only |
| RAID | clean | `train_none.csv` group split by `source_id`; `extra_none.csv` is evaluation-only |

Secondary robustness runs cover M4 monolingual, Deepfake unseen-model/unseen-domain cases, and RAID attacked.
They are not averaged into the primary table unless explicitly labelled.

## Method fidelity

| Method | Locked setting | Local status |
| --- | --- | --- |
| XLM-R fine-tuning | `xlm-roberta-base`; seed 0; LR 2e-5; batch 16; 3 epochs; weight decay 0.01 | runnable on the RTX 5070 Ti; official SemEval training function verified end-to-end |
| Fast-DetectGPT | GPT-J-6B sampling + GPT-Neo-2.7B scoring; fp16; official analytic discrepancy and Gaussian calibration | blocked by 15.9 GiB VRAM; released loader needs about 18.5 GiB on one GPU |
| Binoculars | Falcon-7B observer + Falcon-7B-Instruct performer; bfloat16; 512 tokens; low-FPR threshold | blocked by 15.9 GiB VRAM; released loader needs about 30 GiB on one GPU or two GPUs near 15 GiB each |

No quantization or smaller replacement model is used for the primary table. The Fast-DetectGPT
GPT-Neo-2.7B/GPT-Neo-2.7B configuration is allowed only as a separately labelled official repository variant.

## Metrics and thresholds

Every completed run reports AUROC, AUPRC, accuracy, precision, recall, F1, confusion matrix, and Brier score
when the score is a probability. Three threshold views are stored: the upstream native threshold, fixed 0.5,
and validation-selected F1. The primary paper-faithful command selects `native`; the other views remain in
`metrics.json` for common-protocol comparisons.

## Commands

Run the three primary XLM-R jobs sequentially:

```powershell
& .\baselines\run_xlmr_primary.ps1
```

Run the two paper-faithful zero-shot baselines on hardware that passes the preflight:

```powershell
$python = "C:\Users\three\.conda\envs\TextWave\python.exe"
& $python .\baselines\run.py --dataset all --method fast_detect_gpt --threshold-mode native
& $python .\baselines\run.py --dataset all --method binoculars --threshold-mode native
```

All model/checkpoint outputs are ignored by Git. Experiment manifests retain source identities, split sizes,
method configuration, upstream commit, runtime versions, and the exact fit scope.
