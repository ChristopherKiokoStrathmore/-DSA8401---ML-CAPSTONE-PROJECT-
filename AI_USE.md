# AI use

Phases 1 to 4 of this repository (data pipeline, label splits, baselines, the multi-task training loop, evaluation, synthetic sample, tests, and this note) were drafted by a Cursor cloud agent (Grok 4.7) from the capstone brief, the phase plan, and column-only schemas. The agent did not have the real Airtel workbook, the real label files, or any real customer message. The AfroXLMR T4 run was not executed here.

Phase 5 (the FastAPI service, the TF-IDF smoke fixture, the weight-download script, and the container files) was drafted the same way. The agent did not train AfroXLMR and did not have the checkpoint. The URL and SHA-256 in git are placeholders. Predictions from `make serve-smoke` are a harness check.

Chris Kioko owns the real export, clearance, and the decision to run `make data` on it. Smoke numbers in generated audits are harness checks.
