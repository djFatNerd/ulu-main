#!/usr/bin/env bash
set -euo pipefail

python -m tools.osm.generate_semantic_dataset_large_area \
    40.7580 -73.9855 6000 \
    --max-radius 1500 \
    --resolution 1.0 \
    --output ./times_square_large
