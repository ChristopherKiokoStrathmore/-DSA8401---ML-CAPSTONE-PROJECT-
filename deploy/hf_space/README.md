---
title: Airtel Kenya message classifier
emoji: "📡"
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# DSA 8401 capstone API

`POST /predict` with `{"text": "..."}` returns issue, sentiment, and urgency.

On start the container runs `scripts/download_weights.py`. That downloads `afroxlmr/model.pt` from the Drive folder in `config/weights.json` and checks `artifacts/transformer/weights.sha256`. The file is cached under `/data/afroxlmr` when `/data` is writable, otherwise under `/tmp/afroxlmr`. Weights are not in git.

The Space is free CPU basic (16 GB). The checkpoint is 1.1 GB. One CPU torch process serves it.

If the download fails, the container logs `HARNESS CHECK` and serves the synthetic TF-IDF fixture instead. Those predictions are not the T4 result.
