"""FastAPI app for the three-head classifier.

``GET /health``, ``POST /predict``, ``POST /predict/batch``, and
``POST /predict_batch`` (the path the MULTI-HEAD Next.js app calls).

Every input is passed through :func:`capstone.pii.mask_pii` before the model
sees it. Logs record counts and lengths only, never the message.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from capstone.pii import mask_pii
from capstone.serving.backends import (
    InferenceError,
    LoadedModel,
    ModelNotReady,
    Predictor,
    canonical_labels,
    load_backend,
)
from capstone.serving.schemas import BatchRequest, Health, Prediction, PredictRequest
from capstone.serving.settings import Settings

logger = logging.getLogger("capstone.serving")


def create_app(
    settings: Settings | None = None,
    predictor: Predictor | None = None,
) -> FastAPI:
    """Build an app. Pass ``predictor`` only from tests."""
    settings = settings or Settings.from_env()
    labels = canonical_labels()
    device = "cpu"
    checkpoint = str(settings.checkpoint_dir())
    load_error: str | None = None
    if predictor is None:
        try:
            loaded: LoadedModel = load_backend(
                settings.backend, settings.tfidf_dir, settings.transformer_dir
            )
        except ModelNotReady as exc:
            predictor = None
            load_error = str(exc)
            logger.error("model not ready: %s", load_error)
        else:
            predictor = loaded.predictor
            labels = loaded.labels
            device = loaded.device
            checkpoint = loaded.checkpoint
            logger.info("model ready backend=%s", settings.backend)

    app = FastAPI(
        title="Airtel Kenya message classifier",
        version="0.1.0",
        description=(
            "Issue, sentiment, and urgency for one customer message. "
            "Inputs are masked with the training PII tokens before inference."
        ),
    )
    app.state.settings = settings
    app.state.predictor = predictor
    app.state.load_error = load_error
    app.state.labels = labels
    app.state.device = device
    app.state.checkpoint = checkpoint

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default body echoes `input`, which is the raw message. Drop it.
        logger.info("validation_error path=%s count=%d", request.url.path, len(exc.errors()))
        cleaned = []
        for err in exc.errors():
            cleaned.append(
                {
                    "loc": [str(part) for part in err.get("loc", ())],
                    "msg": str(err.get("msg", "invalid request")),
                    "type": str(err.get("type", "value_error")),
                }
            )
        return JSONResponse(status_code=422, content={"detail": cleaned})

    @app.get("/health", response_model=Health)
    def health(request: Request) -> JSONResponse | Health:
        state = request.app.state
        ready = state.predictor is not None and state.load_error is None
        payload = Health(
            status="ok" if ready else "unavailable",
            model_loaded=ready,
            backend=state.settings.backend,
            device=state.device,
            checkpoint=state.checkpoint,
            issue_classes=len(state.labels["issue"]),
            sent_classes=len(state.labels["sentiment"]),
            urgency_labels=list(state.labels["urgency"]),
            error=None if ready else state.load_error,
        )
        if not ready:
            return JSONResponse(status_code=503, content=payload.model_dump())
        return payload

    @app.post("/predict", response_model=Prediction)
    def predict(body: PredictRequest, request: Request) -> Prediction:
        _check_text(body.text, request.app.state.settings)
        logger.info("predict chars=%d", len(body.text))
        return _predict_many([body.text], request)[0]

    def predict_batch(body: BatchRequest, request: Request) -> list[Prediction]:
        settings = request.app.state.settings
        count = len(body.texts)
        if count == 0:
            raise HTTPException(status_code=422, detail="texts must contain at least one string")
        if count > settings.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"batch exceeds {settings.max_batch_size} texts",
            )
        for text in body.texts:
            _check_text(text, settings)
        longest = max(len(text) for text in body.texts)
        logger.info("predict_batch n=%d max_chars=%d", count, longest)
        return _predict_many(body.texts, request)

    # /predict_batch is the path the Next.js app calls. /predict/batch is the same handler.
    for path in ("/predict/batch", "/predict_batch"):
        app.add_api_route(path, predict_batch, methods=["POST"], response_model=list[Prediction])

    return app


def _check_text(text: str, settings: Settings) -> None:
    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {settings.max_text_chars} characters",
        )


def _predict_many(texts: list[str], request: Request) -> list[Prediction]:
    state = request.app.state
    if state.predictor is None or state.load_error is not None:
        detail = state.load_error or "model unavailable"
        raise HTTPException(status_code=503, detail=detail)
    masked = [mask_pii(text) for text in texts]
    try:
        raw = state.predictor.predict(masked)
    except (ModelNotReady, InferenceError) as exc:
        logger.error("inference unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("inference failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="inference failed") from exc
    if not isinstance(raw, list) or len(raw) != len(texts):
        logger.error("inference returned the wrong number of rows")
        raise HTTPException(
            status_code=503,
            detail="model unavailable: prediction count did not match the batch",
        )
    try:
        return [Prediction.model_validate(item) for item in raw]
    except Exception as exc:
        logger.error("inference output rejected: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="model unavailable: prediction did not match the response schema",
        ) from exc


app = create_app()
