"""Request and response models.

The JSON shape matches ``lib/types.ts`` in the MULTI-HEAD Next.js app and the
live ``POST /predict`` payload that app already calls:

    issue, sentiment, urgency, urgency_score, confidence.issue, confidence.sentiment

``GET /health`` follows the live service fields (``status``, ``model_loaded``,
``device``, ``checkpoint``, ``issue_classes``, ``sent_classes``,
``urgency_labels``) and adds ``backend`` and ``error`` so a failed load is
visible. ``status`` is ``ok`` only when the model loaded.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    texts: list[str]


class Confidence(BaseModel):
    issue: float | None = None
    sentiment: float | None = None


class Prediction(BaseModel):
    issue: str
    sentiment: str
    urgency: str
    urgency_score: float | None = None
    confidence: Confidence | None = None


class Health(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    device: str
    checkpoint: str
    issue_classes: int = Field(description="Count of issue classes, matching the live API.")
    sent_classes: int = Field(description="Count of sentiment classes, matching the live API.")
    urgency_labels: list[str]
    error: str | None = None
