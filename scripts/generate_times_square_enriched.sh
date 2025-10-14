#!/usr/bin/env bash
set -euo pipefail

python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --resolution 1.0 \
    --provider osm_google \
    --output ./times_square_enriched
