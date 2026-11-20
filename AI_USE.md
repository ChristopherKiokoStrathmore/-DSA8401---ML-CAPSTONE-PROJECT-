# AI use

This note is the course disclosure for the capstone. Smoke numbers are harness checks. Real evaluation numbers are copied from the real-data `results.md` and are labelled as such in `report/report.md`.

## People

Chris Kioko labelled the sentiment judgment file (the primary sentiment labels), the 150-row sentiment test, and the urgency gold rows (the 120-row test and the gold rows inside the 868, including the 60-row validation carve). He also retagged the issue file by content. He owns the Airtel export, the clearance decision, and the Colab T4 training run.

## Models and tools

| Tool | What it did | What it did not do |
| --- | --- | --- |
| Claude | Drafting and editing of project text (proposal-side writing and report structure). | Did not see raw customer messages in this repository, and did not train the model. |
| Gemini | Offline silver relabeling written to `gemini_relabeled.csv`. | Excluded from every training, validation, and test pool. The file is loaded only so the 23 exact overlaps with the urgency test can be counted. |
| Groq | Offline distillation of the urgency silver labels. 688 of the 868 urgency training rows are `source=distilled`. | Did not label the urgency gold. Gold rows stay at weight 1. The silver rows are down-weighted and are not the urgency test. |
| Cursor | Code generation by cloud agents: the data pipeline, label splits, baselines, multi-task training loop, evaluation, API, Docker files, tests, the submission cover, and these write-ups. The agents also copied the public MULTI-HEAD Next.js app into `frontend/` and pointed `NEXT_PUBLIC_API_URL` at this FastAPI service. | Did not receive the real workbook or the real label texts. Did not run the T4 training. Did not design the Next.js screens; those come from the MULTI-HEAD repository. |

LLMs were not put in the serving path. Gemini and Groq ran offline, before training, and only to produce silver files. Serving masks text with the same `mask_pii` function used in training and then calls either the TF-IDF pipelines or `load_for_inference`.

## What is not an AI result

The numbers in `report/report.md` and `reports/results.json` were produced by `evaluate.py` on the real test manifests, including the Colab T4 AfroXLMR checkpoint. Harness output from `make smoke` and `make serve-smoke` is not a result.
