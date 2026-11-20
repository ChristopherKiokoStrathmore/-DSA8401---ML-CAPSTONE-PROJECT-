#!/usr/bin/env python3
"""Download AfroXLMR artefacts and verify them against artifacts/transformer/weights.sha256.

Run from the repo root after ``make install``:

    python scripts/download_weights.py

The committed URL is the public Drive folder. The script lists it and
downloads only ``afroxlmr/`` (``model.pt`` and the tokenizer files), then
checks the SHA-256. ``last.pt.tmp`` and ``reports/`` are not fetched. A file
that already matches is skipped. A file URL still works. A placeholder URL
exits 1 before any download. A hash mismatch deletes the partial file and
exits 1. ``WEIGHTS_DIR`` sets the output directory.
"""

from capstone.weights import main

if __name__ == "__main__":
    main()
