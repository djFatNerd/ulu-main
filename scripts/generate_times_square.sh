#!/usr/bin/env bash
set -euo pipefail

python -m tools.osm.generate_semantic_dataset \
    40.7580 -73.9855 1000 \
    --resolution 1.0 \
    --output ./times_square
