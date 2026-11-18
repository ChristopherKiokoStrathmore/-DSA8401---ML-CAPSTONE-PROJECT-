"""DSA 8401 capstone package. Phase 1 is the data pipeline."""

__version__ = "0.1.0"

from capstone.pii import mask_pii

__all__ = ["__version__", "mask_pii"]
