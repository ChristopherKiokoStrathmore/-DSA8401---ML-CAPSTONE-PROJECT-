# DSA 8401 capstone — Airtel Kenya message pipeline

Multi-head classification of Airtel Kenya customer messages: 10 issue classes, binary sentiment (`negative` / `not_negative`), and urgency (`low` / `medium` / `emergency`). Later modelling is AfroXLMR-base with the bottom 8 of 12 layers frozen and a CORN urgency head. This repository currently contains Phase 1 (the workbook pipeline) and Phase 2 (labels and splits).

The real workbook and the real label files are not in git. They stay in `data/private/` on a machine that has clearance. `data/sample/` is fictional and was produced with seed 42.

## Layout

```
src/capstone/pii.py            mask_pii, shared by training prep and serving
src/capstone/prepare_data.py   workbook -> parquet + aggregate audit
src/capstone/duplicates.py     exact and near-duplicate group ids
src/capstone/labels.py         the seven label filenames and their columns
src/capstone/splits.py         grouped splits, gold removal, manifests
src/capstone/agreement.py      hand vs platform sentiment agreement
src/capstone/leakage.py        test vs train/val duplicate check
data/sample/messages.csv       synthetic workbook
data/sample/labels/            synthetic label files, already masked
docs/label_rubric.md           class names, provenance, agreement
data/private/                  git-ignored real output
reports/                       generated audits and manifests, git-ignored
```

## Setup

Python 3.11 or newer. From the repo root:

```bash
make install
make lint
make test
make smoke
```

`make data DATA=/path/to/workbook.xlsx` runs the workbook pipeline on a real export and writes `data/private/messages.parquet`.

`make labels LABELS=/path/to/label_dir` builds manifests, recomputes agreement, and runs the leakage check. The directory must already use the filenames in `docs/label_rubric.md`. Copy real exports into that directory under those names. Do not commit them.

Numbers from `make smoke` are a harness check, not results.

## What the pipeline does

1. Reads the workbook once (`.xlsx` or `.csv`).
2. Fails if any required column is missing: `Channel`, `Type`, `Posted on`, `Text`, `Last action`, `Action date`, `Sentiment`, `Tags`, `Notes`.
3. Counts `Unnamed: N` overflow columns and any unexpected column, then drops them.
4. Keeps `Channel` and `Type` beside `Channel_norm` and `Type_norm`. `Message` and `message` collapse. The Airtel page variants, including the parenthetical phone forms, collapse to `airtel_kenya`. `'@AIRTEL_KE` becomes `airtel_ke`, `'@airtel_kenya` becomes `airtel_kenya_handle`, and `Airtel Kenya Twitter Search` becomes `airtel_kenya_twitter_search`.
5. Masks PII in `Text`, `Tags`, and `Notes`. The raw string is not written to parquet.
6. Drops empty texts, non-linguistic texts (no letters: emoji, punctuation, digits, USSD), and auto receipts. Each rule's count is in the audit. The first match wins.
7. Assigns `dup_group_id` on the kept rows.

Dates outside 2025-11-01 .. 2026-04-17 are counted in the audit and kept. The platform `Sentiment` column is not the hand label.

## PII tokens

`mask_pii` emits only the four placeholders already used in the labelled files:

| Token | What it replaces |
| --- | --- |
| `[PHONE]` | Kenyan numbers: `07xx` / `01xx`, `+254` / `254`, optional trunk `(0)`, spaces, dashes, dots, slashes |
| `[URL]` | `http://`, `https://`, `www.` |
| `[USER]` | emails and `@handles` |
| `[ID]` | labelled national IDs (including `kitambulisho`), labelled account / paybill / till numbers, M-Pesa-like confirmation codes |

There is no `[EMAIL]` token. Emails are `[USER]` so serving matches training. A second call is a no-op on text that already uses these tokens. Prices, dates, `*544#`, and bundle sizes such as `1GB` are not masked. Bare 7–8 digit numbers are not treated as IDs unless a label sits in front of them.

## Near-duplicate groups

Exact duplicates are collapsed on the lowercased, whitespace-collapsed masked text. Near-duplicates are then found with MinHash LSH (character 3-grams, 32 permutations, 8 bands of 4, seed 42) plus shared 6- and 12-character prefixes. A candidate pair is kept when the Dice coefficient of those 3-gram sets is at least 0.80. The shared `dup_group_id` is the smallest source `row_id` in the component, so later splits can group on one column and a leakage check can reject any shared id. Buckets over 200 strings are length-striped and windowed so a few hundred thousand short rows stay on one CPU.

## Labels and splits

`docs/label_rubric.md` is the class list, the provenance of each file, and the agreement table. The ten issue names match `eval/README.md` in the MULTI-HEAD repo. Urgency `low` / `medium` / `emergency` matches `lib/urgency.ts`.

Gold is the 120-row urgency test plus the 150-row sentiment test. Exact and near-duplicate matches of that gold are removed from every train and validation pool. Gemini is not used for training. Issue is a grouped 70/15/15 split, seed 42. Sentiment train and validation come from the hand file; the 150 is the test. Urgency validation is about 60 gold rows carved from the 868, and the urgency test stays the fixed 120. Manifests record row ids, duplicate group ids, and the sha256 of each source file. They do not record message text.

`tests/test_no_leakage.py` builds those manifests from the sample and fails if any exact or near-duplicate sits in both test and train/val. `python -m capstone.leakage --manifests <dir>` is the same check for a real directory.

## Models and evaluation

Baselines, fitted on the training split with `C` chosen on validation only:

- majority class per head
- an emergency keyword heuristic (English, Swahili, and Sheng substrings; other heads fall back to keywords or the majority class)
- word and character TF-IDF plus class-balanced logistic regression, one pipeline per head

The TF-IDF files are `artifacts/tfidf/{issue,sentiment,urgency}.joblib` plus `artifacts/tfidf/labels.json`. Each joblib is a scikit-learn Pipeline whose `predict_proba` masks PII and then vectorises. That is the format the API loads.

The multi-task model is `Davlan/afro-xlmr-base`. Embeddings and the bottom 8 of 12 layers stay frozen. The heads are issue softmax (10), sentiment softmax (2), and urgency CORN (3 ordinal levels, 2 conditional logits). A row only contributes loss for the heads it has labels for. Class weights and the urgency row weights are applied. One seed, 42. Checkpoints are written every epoch and `--resume` continues from `artifacts/afroxlmr/checkpoints/last.pt`. Early stopping watches the mean of validation issue macro-F1, sentiment macro-F1, and urgency QWK. The run is sized for one Colab T4 in fp16, about 60–90 minutes. The training log prints an ETA after the first step.

`capstone.models.multitask.load_for_inference(directory).predict(texts)` returns, per text, the issue label and probabilities, the sentiment label and probability, and the urgency level, probabilities, and emergency flag. It runs on CPU. `predict` masks PII first.

The emergency flag uses a threshold chosen on the urgency validation split only: maximise emergency recall subject to a false-emergency rate of at most 0.20. That threshold is applied once to the test split. `evaluate` writes `reports/results.json` and `reports/results.md` (macro-F1, per-class precision/recall/F1 with support, confusion matrices, urgency QWK, emergency recall, false-emergency rate, 95% bootstrap intervals, and paired intervals against TF-IDF). Mistakes go to `data/private/errors.csv` with a `category_hint` column. Thirty of them are marked `taxonomy_pick` for the error review. That file is git-ignored.

```bash
make train-baselines LABELS=data/sample/labels MANIFESTS=reports/manifests KIND=harness
make train-transformer LABELS=... MANIFESTS=... CONFIG=configs/afroxlmr.yaml
make evaluate LABELS=... MANIFESTS=... KIND=data
make real DATA=/path/to/workbook.xlsx LABELS=/path/to/label_dir
```

`make smoke` fits the baselines, trains the tiny random encoder for one step, and evaluates the sample. Those numbers are a harness check. `colab/run_real.ipynb` is the T4 path: clone, install, upload the workbook and the seven label files, `make real`, download the report and the weights. The notebook has no saved outputs. The T4 run itself is not executed in this repository checkout.

## Privacy

Do not commit `.xlsx`, `.csv`, or `.parquet` outside `data/sample/`. A test fails the suite if git is tracking one. Do not commit `.env*`, `__pycache__/`, `node_modules/`, or `.next/`. The audit prints counts only.
