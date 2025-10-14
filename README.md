# Semantic Map Dataset Toolkit

This repository contains a single-purpose toolkit for generating 1 m semantic
rasters and hierarchical building metadata from OpenStreetMap (OSM). It focuses
exclusively on the components required for offline semantic dataset creation.

## Contents

- `tools/scripts/generate_semantic_dataset.py` – Command-line script that queries
  OSM/Overpass, rasterizes semantic maps, and exports building taxonomies.
- `tools/scripts/generate_semantic_dataset_large_area.py` – Helper CLI that
  tiles large queries into manageable chunks, merges outputs, and reports
  per-tile progress.
- `docs/generate_semantic_dataset.md` – Detailed usage guide describing inputs,
  outputs, dependencies, and scaling recommendations.
- `docs/generate_semantic_dataset_large_area.md` – Companion guide for the
  large-area tiling workflow.
- `requirements.txt` – Minimal Python dependencies needed to run the script.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate a dataset tile by providing latitude, longitude, and radius (meters):
   ```bash
   python tools/scripts/generate_semantic_dataset.py \
       40.7580 -73.9855 1000 \
       --resolution 1.0 \
       --output ./times_square
   ```
3. Review the outputs in the chosen directory. See the documentation in
   `docs/generate_semantic_dataset.md` for advanced configuration and data
   schema details.

## License

This repository redistributes only original code authored for the semantic
dataset workflow. Refer to the repository license for additional details.
