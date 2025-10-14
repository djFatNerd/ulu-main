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
2. Choose the script that matches your area of interest:
   - For **single tiles or modest regions** (that fit within Overpass limits),
     run `generate_semantic_dataset.py` directly, or execute the convenience
     wrapper:
       ```bash
       # direct Python invocation
       python tools/scripts/generate_semantic_dataset.py \
           40.7580 -73.9855 1000 \
           --resolution 1.0 \
           --output ./times_square

       # or call the bundled helper script
       ./scripts/generate_times_square.sh
       ```
   - For **very large regions** that may exceed API or memory limits, run the
     tiling helper `generate_semantic_dataset_large_area.py`. It does **not**
     run automatically—invoke it explicitly when you know the query is large.
     A helper shell script is also provided:
       ```bash
       # direct Python invocation
       python tools/scripts/generate_semantic_dataset_large_area.py \
           40.7580 -73.9855 6000 \
           --max-radius 1500 \
           --resolution 1.0 \
           --output ./times_square_large

       # or call the bundled helper script
       ./scripts/generate_times_square_large.sh
       ```
3. Review the outputs in the chosen directory. The standard script documentation
   lives in `docs/generate_semantic_dataset.md`. The tiling workflow and
   combined outputs are described in
   `docs/generate_semantic_dataset_large_area.md`.

## License

This repository redistributes only original code authored for the semantic
dataset workflow. Refer to the repository license for additional details.
