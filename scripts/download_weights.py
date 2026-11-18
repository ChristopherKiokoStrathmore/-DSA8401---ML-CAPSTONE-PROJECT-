#!/usr/bin/env python3
"""Download AfroXLMR artefacts and verify them against artifacts/transformer/weights.sha256.

Run from the repo root after ``make install``:

    python scripts/download_weights.py
    AFROXLMR_WEIGHTS_URL=<public-drive-url> python scripts/download_weights.py

The committed URL and digest are placeholders. A placeholder URL exits 1 before
any download. A hash mismatch deletes the partial file and exits 1.
"""

from capstone.weights import main

if __name__ == "__main__":
    main()
