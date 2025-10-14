# Semantic Map Dataset Toolkit

This repository contains a single-purpose toolkit for generating 1 m semantic
rasters and hierarchical building metadata from OpenStreetMap (OSM). It is a
minimal extraction of the original VIRL project, retaining only the components
required for offline semantic dataset creation.

## Contents

- `tools/scripts/generate_semantic_dataset.py` – Command-line script that queries
  OSM/Overpass, rasterizes semantic maps, and exports building taxonomies.
- `docs/generate_semantic_dataset.md` – Detailed usage guide describing inputs,
  outputs, dependencies, and scaling recommendations.
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
dataset workflow. Consult the upstream VIRL project for licensing terms related
to any remaining files.
