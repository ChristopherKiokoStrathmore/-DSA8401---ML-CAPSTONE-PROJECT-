"""HTTP serving for the three-head classifier.

Phase 3 owns the trained models. This package only loads them.

TF-IDF artefacts, written by Phase 3 (or by the smoke fixture):

    artifacts/tfidf/issue.joblib
    artifacts/tfidf/sentiment.joblib
    artifacts/tfidf/urgency.joblib
    artifacts/tfidf/labels.json

Each joblib is a scikit-learn Pipeline with ``predict_proba``. ``labels.json``
maps ``issue``, ``sentiment``, and ``urgency`` to class lists. Urgency order
is ``low``, ``medium``, ``emergency``.

Transformer artefacts live in a directory passed to
``capstone.models.multitask.load_for_inference``. That import is lazy. If the
module is missing or loading fails, ``GET /health`` reports it and the predict
routes return 503.
"""
