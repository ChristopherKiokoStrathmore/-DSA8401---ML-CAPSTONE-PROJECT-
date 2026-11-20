# Deploy

The live service is Railway Hobby. The cover still says `LIVE_DEMO_URL` until a domain is generated. This checkout does not deploy it. NOT VERIFIED.

## Railway

`railway.json` uses the root `Dockerfile`, healthchecks `/health` with a 900 second timeout (model load plus the first-start download), and restarts on failure. The process binds `0.0.0.0:$PORT`. Compose leaves `PORT` at 8000. Railway sets `PORT` itself.

The container needs about 3 GB of RAM. `model.pt` is 1.1 GB, and one CPU torch process loads it.

1. New Project.
2. Deploy from this GitHub repository.
3. Add a Volume mounted at `/data`.
4. Set variables:
   - `MODEL_BACKEND=transformer`
   - `WEIGHTS_DIR=/data`
   - `CORS_ORIGINS=*` (or the Vercel origin)
5. Generate Domain.
6. Wait for `/health` to become `ok`. The first start lists the Drive folder and downloads only `afroxlmr/` (`model.pt` and the tokenizer files), then checks the SHA-256. `last.pt.tmp` and `reports/` are not fetched. A later start skips the download when `model.pt` already matches.

```bash
curl -s "https://<railway-domain>/health"
curl -s -X POST "https://<railway-domain>/predict" \
  -H 'content-type: application/json' \
  -d '{"text":"My line has been down since morning, please help"}'
```

### Frontend

Vercel: import this repository, set the root directory to `frontend`, set `NEXT_PUBLIC_API_URL` to the Railway URL with no `/predict` suffix, and redeploy.

Or add a second Railway service from the same repo with root directory `frontend` and the same `NEXT_PUBLIC_API_URL`.

## Hugging Face Space (optional, needs PRO)

A Docker Space on the free plan returns HTTP 402. `deploy/hf_space/` remains for a PRO account. `make hf-space` still creates public `Kioko1/airtel-multihead` (`cpu-basic`) when `HF_TOKEN` is set, using only `pip install huggingface_hub`. The direct URL would be https://kioko1-airtel-multihead.hf.space. That upload is NOT VERIFIED.
