#!/usr/bin/env python3
"""Download AfroXLMR artefacts and verify them against artifacts/transformer/weights.sha256.

Run from the repo root after ``make install``:

    python scripts/download_weights.py

The committed URL is the public Drive folder. The script downloads that folder
with gdown and keeps ``afroxlmr/model.pt``, then checks the SHA-256. A file
URL still works. A placeholder URL exits 1 before any download. A hash
mismatch deletes the partial file and exits 1.
"""

from capstone.weights import main

if __name__ == "__main__":
    main()
