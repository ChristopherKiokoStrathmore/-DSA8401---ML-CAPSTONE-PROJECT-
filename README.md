# DSA 8401 capstone — Airtel Kenya message classifier

Three decisions for one customer message: the issue (10 classes), the sentiment (`negative` or `not_negative`), and the urgency (`low`, `medium`, or `emergency`). The messages mix English, Swahili, and Sheng. The model is a shared AfroXLMR-base encoder with three heads. A TF-IDF logistic regression serves the same API when no GPU is available.

![End-to-end architecture: data pipeline, modelling, and the Docker API](docs/architecture.png)

Real customer text is not in this repository. It stays in git-ignored `data/private/`. Clearance status: cleared for coursework. The write-up is `report/report.md`. Technique choices are in `docs/technique_justification.md`. Who labelled what, and which tools wrote code, is in `AI_USE.md`.

## Setup

Python 3.11 or newer. From the repo root:

```bash
make install
make lint
make test
make smoke
make serve-smoke
```

`make smoke` runs the pipeline, the baselines, a one-step random encoder, and evaluation on `data/sample/`. Every number it prints is a harness check, not a result.

## Reproduction

Sample, from a clean checkout:

```bash
make install
make smoke
make serve-smoke
```

Real data, on a machine that has the export. Rename the label files to the loader names in `docs/label_rubric.md` (`FILE_MAP` in `src/capstone/labels.py` lists the original names, including `sentiment_test_to_annotate(in).csv`, `urgency_v3_train (2).csv`, and `urgency_v3_test (2).csv`).

```bash
make real DATA=/path/to/workbook.xlsx LABELS=/path/to/label_dir
```

That runs prepare, splits, baselines, AfroXLMR training, and evaluation. Outputs land in `data/private/`, `reports/`, and `artifacts/`. Do not commit them.

Colab T4: open `colab/run_real.ipynb`. It clones `main`, installs, uploads a labels-and-manifests bundle, fits baselines, trains, evaluates, prints the SHA-256, and zips the report and the weights. The notebook stores no outputs. The T4 run is not executed in this checkout.

## API

`MODEL_BACKEND=tfidf` (the default) loads `artifacts/tfidf/{issue,sentiment,urgency}.joblib` and `artifacts/tfidf/labels.json`. Those joblibs are scikit-learn pipelines: mask, word and character TF-IDF, logistic regression, `predict_proba`.

`MODEL_BACKEND=transformer` calls `capstone.models.multitask.load_for_inference`. The default directory is `artifacts/afroxlmr`. Override it with `TRANSFORMER_DIR`. Each `predict` row has the issue label and probabilities, the sentiment label and probability, and the urgency level, probabilities, and emergency flag. The service maps that onto the JSON below. A missing checkpoint stays up: `/health` is 503 and says why.

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"My line has been down since morning, please help"}'
curl -s -X POST http://127.0.0.1:8000/predict_batch \
  -H 'content-type: application/json' \
  -d '{"texts":["bundle haijaingia","asante"]}'
```

`/predict/batch` is the same handler as `/predict_batch`. A prediction is:

```json
{
  "issue": "Network_Issues",
  "sentiment": "negative",
  "urgency": "medium",
  "urgency_score": 0.42,
  "confidence": {"issue": 0.81, "sentiment": 0.77}
}
```

Every input goes through `mask_pii` before the model sees it. The same function masks training text. Logs record counts and lengths, not the message. Text longer than `MAX_TEXT_CHARS` (default 4000) or a batch larger than `MAX_BATCH_SIZE` (default 32) is HTTP 422, and the body does not echo the message.

CORS allows `https://multi-head.vercel.app`, any `https://*.vercel.app` origin, and local `http://localhost` or `http://127.0.0.1` on any port.

## Docker

```bash
docker compose up --build
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"My line has been down since morning, please help"}'
```

The image is Python slim, installs a CPU torch wheel, and runs uvicorn as a non-root user. `MODEL_BACKEND` defaults to `tfidf`. `HEALTHCHECK` calls `/health`. If the TF-IDF directory is empty at start, the entrypoint trains the synthetic fixture so the container can answer. Mount the Phase 3 `artifacts/tfidf` directory to serve the real pipelines instead.

## Weights

AfroXLMR weights are not in git. `config/weights.json` holds a placeholder Google Drive URL. `artifacts/transformer/weights.sha256` holds a placeholder digest. Training also writes `artifacts/afroxlmr/sha256.txt` for the state dict it just saved.

```bash
AFROXLMR_WEIGHTS_URL='https://drive.google.com/file/d/<id>/view' python scripts/download_weights.py
```

`WEIGHTS_URL` is an alias. A Drive link uses gdown. Any other URL uses urllib. The script checks the file against the SHA-256 and deletes a mismatch. The placeholder URL is refused before any download.

## Layout

```
src/capstone/prepare_data.py    workbook to parquet and an aggregate audit
src/capstone/pii.py             mask_pii, shared by training and serving
src/capstone/duplicates.py      exact and near-duplicate group ids
src/capstone/labels.py          FILE_MAP and the seven canonical filenames
src/capstone/splits.py          grouped splits, gold removal, manifests
src/capstone/models/            baselines, CORN, AfroXLMR multi-task
src/capstone/evaluate.py        test metrics, bootstrap, error file
src/capstone/serving/           FastAPI app
src/capstone/weights.py         download and SHA-256 check
docs/architecture.png           diagram above
docs/label_rubric.md            classes, provenance, recomputed agreement
docs/technique_justification.md one row per syllabus area
report/report.md                the capstone report
colab/run_real.ipynb            T4 notebook, no stored outputs
data/sample/                    fictional workbook and labels, seed 42
data/private/                   git-ignored real data
```

## Pipeline notes

`prepare_data` fails if a required column is missing, keeps original channel and type beside the normalised values, counts and drops `Unnamed:` overflow columns, and assigns `dup_group_id`. Dates outside 2025-11-01 through 2026-04-17 are counted and kept. The platform `Sentiment` column is not the hand label.

Near-duplicates use character 3-gram MinHash (32 permutations, 8 bands of 4, seed 42) plus shared prefixes, confirmed at Dice 0.80. Training examples are one per source row id. Two rows that normalise to the same string both stay, so a label is not dropped.

The emergency threshold is chosen on the urgency validation split only: maximise recall subject to a false-emergency rate of at most 0.20, then apply it once to test. `evaluate.py` skips a model whose artefact is missing and records `not evaluated` in `results.json`. Mistakes go to `data/private/errors.csv` with a category hint. Thirty are marked for the taxonomy. That file is git-ignored.

Do not commit `.xlsx`, `.csv`, or `.parquet` outside `data/sample/`. Do not commit `.env*`, `__pycache__/`, `node_modules/`, `.next/`, or model weights.
