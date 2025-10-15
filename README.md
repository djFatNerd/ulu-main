# Semantic Map Dataset Toolkit

This repository contains a single-purpose toolkit for generating 1 m semantic
rasters and hierarchical building metadata from OpenStreetMap (OSM). It focuses
exclusively on the components required for offline semantic dataset creation.

## Contents

- `tools/osm/generate_semantic_dataset.py` – Command-line script that queries
  OSM/Overpass, rasterizes semantic maps, and exports building taxonomies.
- `tools/osm/generate_semantic_dataset_large_area.py` – Helper CLI that
  tiles large queries into manageable chunks, merges outputs, and reports
  per-tile progress.
- `tools/multisource/generate_semantic_dataset_enriched.py` – Multisource CLI
  that reuses the OSM rasterization pipeline and enriches building metadata
  with external providers such as Google Places.
- `docs/generate_semantic_dataset.md` – Detailed usage guide describing inputs,
  outputs, dependencies, and scaling recommendations.
- `docs/generate_semantic_dataset_large_area.md` – Companion guide for the
  large-area tiling workflow.
- `docs/multisource/guide.md` – Multisource enrichment workflow, rate limits,
  privacy guidance, and attribution requirements.
- `requirements.txt` – Minimal Python dependencies needed to run the script.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Choose the script that matches your area of interest:
   - For **single tiles or modest regions** (that fit within Overpass limits),
     run `generate_semantic_dataset.py` as a module, or execute the convenience
     wrapper:
       ```bash
       # direct Python invocation
       python -m tools.osm.generate_semantic_dataset \
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
       python -m tools.osm.generate_semantic_dataset_large_area \
           40.7580 -73.9855 6000 \
           --max-radius 1500 \
           --resolution 1.0 \
           --output ./times_square_large

       # or call the bundled helper script
       ./scripts/generate_times_square_large.sh
       ```
3. Review the outputs in the chosen directory. The standard script documentation
   lives in `docs/osm/generate_semantic_dataset.md`. The tiling workflow and
   combined outputs are described in
   `docs/osm/generate_semantic_dataset_large_area.md`.

### One-command reference

Use the following one-liners when you just need to trigger a specific workflow:

| Workflow | Command |
| --- | --- |
| OSM single-area | `python -m tools.osm.generate_semantic_dataset 40.7580 -73.9855 1000 --resolution 1.0 --output ./times_square` |
| OSM large-area tiling | `python -m tools.osm.generate_semantic_dataset_large_area 40.7580 -73.9855 6000 --max-radius 1500 --resolution 1.0 --output ./times_square_large` |
| Multisource enrichment | `python tools/multisource/generate_semantic_dataset_enriched.py 40.7580 -73.9855 1000 --provider osm_google --output ./times_square_enriched` |
| Free-tier preset | `FREE_TIER=true ./scripts/multi-resource.sh 40.7580 -73.9855 1000 ./times_square_free` |

## Multisource enrichment workflow

When you need provider-sourced attributes (official names, Google category
hierarchies, ratings, opening hours), run the multisource CLI instead of the
OSM-only scripts:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --provider osm_google \
    --output ./times_square_enriched
```

This script shares the same rasterization pipeline but augments the building
GeoJSON with provider-specific fields and provenance flags. See
`docs/multisource/guide.md` for rate limiting, privacy guidance, and examples of
switching between OSM-only, Google-only, and hybrid modes. Unlike the original
workflow, the enriched CLI stores provider responses (including IDs and match
confidence) so downstream users can trace the origin of each attribute.

To combine multiple enrichment sources—such as Google Places, government
GeoJSON catalogs, and cached CSV registries—pass the new `--providers` flag and
corresponding data-field overrides. The CLI merges metadata sequentially while
tracking field-level provenance:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --providers google local_geojson local_csv \
    --local-geojson ./data/city_open_data.geojson \
    --local-geojson-name-field official_name \
    --local-csv ./data/business_registry.csv \
    --local-csv-category-fields naics_description \
    --output ./times_square_multi
```

If you prefer a one-liner, use the bundled helper script (remember to export a
`GOOGLE_MAPS_API_KEY` environment variable when required):

```bash
./scripts/generate_times_square_enriched.sh
```

For more granular control, use the turnkey helper script
`./scripts/multi-resource.sh`. It wraps the enriched CLI and requires three
positional arguments: latitude, longitude, and radius (in meters). An optional
fourth argument overrides the output directory (defaults to
`./semantic_dataset_multi`). For example:

```bash
# LAT       LON        RADIUS(m)  OUTPUT_DIR
./scripts/multi-resource.sh 40.7580 -73.9855 1000 ./times_square_multi
```

Provider-specific inputs are toggled via environment variables. Combine them in
front of the command when needed:

```bash
ENABLE_GOOGLE=true \
LOCAL_GEOJSON_PATH=./data/city_open_data.geojson \
LOCAL_CSV_PATH=./data/licensed_businesses.csv \
./scripts/multi-resource.sh 40.7580 -73.9855 1000 ./times_square_multi
```

### Zero-cost, open-data preset

To experiment without any paid APIs, enable the new free-tier preset. It
activates OSM-only labeling and automatically wires in the sample open datasets
under `data/open/` (feel free to swap them with your own Microsoft building
footprints, Mapillary exports, or municipal open data downloads):

```bash
./scripts/multi-resource.sh 40.7580 -73.9855 1000 ./times_square_free \
    FREE_TIER=true
```

Behind the scenes this forwards `--free-tier` to the multisource CLI, forcing an
OSM + offline-provider workflow. You can override the bundled datasets by
passing `FREE_TIER_GEOJSON=/path/to/file.geojson` or
`FREE_TIER_CSV=/path/to/file.csv` alongside the command.

If you run the script without the required positional arguments it will print a
usage message and exit immediately. The helper also validates local file paths,
applies shared throttling parameters, and automatically constructs the
appropriate `--providers` invocation for the Python entry point.

## License

This repository redistributes only original code authored for the semantic
dataset workflow. Refer to the repository license for additional details.
