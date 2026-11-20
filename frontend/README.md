# Frontend

Next.js demo copied from [MULTI-HEAD-](https://github.com/ChristopherKiokoStrathmore/MULTI-HEAD-). It has no model of its own.

`lib/api.ts` reads `NEXT_PUBLIC_API_URL` and calls this repository's FastAPI service:

- `GET /health`
- `POST /predict` with `{"text": "..."}`
- `POST /predict_batch` with `{"texts": ["...", "..."]}`

Each prediction is `issue`, `sentiment`, `urgency`, `urgency_score`, and `confidence.issue` / `confidence.sentiment`, matching `lib/types.ts` and `capstone.serving.schemas.Prediction`. Batch chunks are 20 texts. The API default limit is 32.

```bash
cp .env.example .env.local
npm ci
npm run dev
```

See `../deploy/README.md` for the Modal URL and the Vercel root directory (`frontend`).
