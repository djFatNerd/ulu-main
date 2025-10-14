#!/usr/bin/env bash
set -euo pipefail

python tools/scripts/generate_semantic_dataset.py \
    40.7580 -73.9855 1000 \
    --resolution 1.0 \
    --output ./times_square
