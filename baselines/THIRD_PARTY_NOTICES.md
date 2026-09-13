# Third-party provenance notices

## PAN'25 TF-IDF-SVM

The baseline design is derived from
`pan-webis-de/pan25-generative-ai-authorship-verification`, copyright 2025 Janek Bevendorff, Webis, licensed under
Apache License 2.0. This repository reimplements the small estimator configuration and does not vendor the
upstream model pickle or source tree. The upstream license is available at
<https://github.com/pan-webis-de/pan25-generative-ai-authorship-verification/blob/main/LICENSE>.

## BertAA stylometric features

The feature definitions and ordering were verified against the public notebook linked by Fabien et al. (2020),
“BertAA: BERT fine-tuning for Authorship Attribution.” The notebook does not state a software license. No notebook
code is redistributed here; the implementation is a clean-room behavioral reimplementation of the published
feature equations and estimator settings.
