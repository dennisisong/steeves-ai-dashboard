#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="ALY6980 (2).ipynb"
OUTPUT_NOTEBOOK="ALY6980 (2)-executed.ipynb"

python3 -m nbconvert \
  --to notebook \
  --execute \
  --Application.log_level=ERROR \
  "$NOTEBOOK" \
  --output "$OUTPUT_NOTEBOOK"

echo "Notebook executed successfully: $OUTPUT_NOTEBOOK"
