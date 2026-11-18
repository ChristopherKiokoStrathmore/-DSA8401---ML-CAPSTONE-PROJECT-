"""DSA 8401 capstone package.

Phase 1 is the data pipeline, Phase 2 is labels and splits, and Phase 5 is
the serving API in ``capstone.serving``. Importing this package does not load
a model.
"""

__version__ = "0.1.0"

from capstone.pii import mask_pii

__all__ = ["__version__", "mask_pii"]
