# Hugging Face Space

NOT VERIFIED. This checkout has no Hugging Face token, so the Space was not created and nothing was uploaded. The live demo URL on the cover is still `LIVE_DEMO_URL`.

The image listens on port 7860. A free CPU basic Space has 16 GB of RAM. `model.pt` is 1,112,288,438 bytes (about 1.1 GB). The Dockerfile installs the CPU torch wheel, sets one thread for the numeric libraries, and runs a single process, so the footprint stays inside that 16 GB. The loaded transformer was not measured here.

From a fresh clone, install the Hub client and publish. The script creates `Kioko1/airtel-multihead` when it is missing (`create_repo`, `repo_type="space"`, `space_sdk="docker"`, `private=False`, `exist_ok=True`, hardware `cpu-basic`), then `upload_folder` of the staged tree. `HF_TOKEN` is read from the environment and is not placed on the command line.

```bash
pip install huggingface_hub
export HF_TOKEN=hf_...
make hf-space
```

`HF_SPACE` overrides the repo id. The default is `Kioko1/airtel-multihead`. `make hf-space` runs `python3 scripts/push_hf_space.py`, which stages `deploy/hf_space` plus `src`, `config`, `scripts`, `data/sample`, and `artifacts/transformer/weights.sha256`.

The running app is https://kioko1-airtel-multihead.hf.space

```bash
curl -s -X POST "https://kioko1-airtel-multihead.hf.space/predict" \
  -H 'content-type: application/json' \
  -d '{"text":"My line has been down since morning, please help"}'
```

The first start downloads `afroxlmr/model.pt` from the Drive folder in `config/weights.json` and checks the SHA-256. A writable `/data` keeps the file across restarts. Otherwise it is stored under `/tmp`. Weights are not in git.

Vercel:

1. Import this GitHub repository.
2. Set the root directory to `frontend`.
3. Set `NEXT_PUBLIC_API_URL` to `https://kioko1-airtel-multihead.hf.space` with no `/predict` suffix.
4. Redeploy. `NEXT_PUBLIC_*` is baked in at build time.

`CORS_ORIGINS` defaults to `*`.

A local `docker build` of this image, run with a weights URL that cannot be fetched, serves the synthetic TF-IDF fixture. That curl is a harness check, not the T4 model. The Space upload itself is NOT VERIFIED.
