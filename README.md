# Semantic Map Dataset Toolkit

The toolkit assembles building-level semantic datasets by combining OpenStreetMap
footprints with optional Overture Maps, Google Places, and local offline data.
Each run ingests a bounding box defined by latitude, longitude, radius, and
raster resolution, then enriches every OSM footprint with the providers you
select. The main Python entrypoint is `tools/multisource/generate_semantic_dataset_enriched.py`.

## Installation

Install dependencies once before running any scripts:

```bash
pip install -r requirements.txt
```

## Data providers

| Provider | What you get | Access method | Credentials |
| --- | --- | --- | --- |
| **OpenStreetMap (OSM)** | Public building footprints, roads, tags | Overpass API (`OVERPASS_URL`, optional `FALLBACK_OVERPASS`) | none |
| **Overture Maps** | Community-maintained place and building attributes | [`overturemaps` CLI](https://pypi.org/project/overturemaps/) | none |
| **Google Places API** | Commercial POI metadata (names, categories, ratings, hours, phones, URLs) | Google Cloud Places API | `GOOGLE_MAPS_API_KEY` or `GOOGLE_API_KEY` |
| **Local data** | Your own GeoJSON/CSV inventories | Command-line flags on the Python entrypoint | managed by you |

### Provider notes
- **OSM:** Heavy usage should respect Overpass fair-use policies. Override
  `REQUEST_SLEEP` to stagger requests or host a private Overpass instance.
- **Overture Maps:** Install the `overturemaps` package in the active Python
  environment. Downloads are free and cached locally (see `--overture-cache-dir`).
- **Google Places:** Enable the Places APIs in Google Cloud, create an API key,
  and supply it via environment variable or `--google-api-key`. Google requires a
  billing account; monthly free credits apply before usage is billed.
- **Local data:** Provide custom GeoJSON/CSV through
  `--local-geojson` / `--local-csv` when calling the Python entrypoint.

## Running the scripted workflows

Each shell script in `scripts/` accepts the same positional arguments:

```bash
./scripts/<script_name>.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
```

- `LAT`, `LON`: decimal degrees for the query centre.
- `RADIUS`: half the side length of the square in metres.
- `RESOLUTION`: optional raster resolution in metres (default `1.0`).
- `OUTPUT_DIR`: optional destination directory (defaults per workflow).

Environment overrides shared across scripts include `MATCH_DISTANCE`,
`PROVIDER_RADIUS`, `REQUEST_SLEEP`, `LOG_LEVEL`, `OVERPASS_URL`, and
`FALLBACK_OVERPASS`. Provider-specific options (e.g. `OVERTURE_THEME`,
`OVERTURE_CACHE_DIR`) are passed through to the underlying tooling.

### Mode quick reference

| Workflow | Command | Providers merged | Credentials required |
| --- | --- | --- | --- |
| OSM only | `./scripts/run_osm.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]` | OSM | none |
| OSM + Overture (free stack) | `./scripts/run_osm_overture.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]` | OSM + Overture | `overturemaps` package |
| OSM + Overture (raw payloads) | `./scripts/run_osm_overture_full.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]` | OSM + Overture | `overturemaps` package |
| OSM + Overture + Google (full details) | `./scripts/run_osm_overture_google.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]` | OSM + Overture + Google Places | `GOOGLE_MAPS_API_KEY` (or `GOOGLE_API_KEY`), `overturemaps` package |
| OSM + Overture + Google (cost-down) | `./scripts/run_osm_overture_google.sh --cost-down LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]` | OSM + Overture + Google Places (budget-optimised) | `GOOGLE_MAPS_API_KEY` (or `GOOGLE_API_KEY`), `overturemaps` package |

The specialised scripts (`run_google_full.sh`, `run_google_cost_down.sh`, `run_osm_google.sh`, etc.) remain available for
provider-specific debugging or legacy automation, but the table above covers the
recommended combinations.

### Usage examples

```bash
# OSM-only dataset
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm

# OSM + Overture (free enrichment)
./scripts/run_osm_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture

# OSM + Overture with full raw payloads and minimal filtering
./scripts/run_osm_overture_full.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_full

# Full Google enrichment (all fields)
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google

# Cost-down Google enrichment (minimal paid calls)
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_overture_google.sh --cost-down 40.7580 -73.9855 1000 1.0 \
  ./semantic_dataset_osm_overture_google_cost_down
```

### Batch processing multiple cities

Use `./scripts/run_osm_overture_full_batch.sh` to iterate over a list of target
cities stored in `docs/cities/cities.jsonl`. On Windows, run the command from a
shell that provides `bash` (for example Git Bash or WSL) so the underlying
workflow script can execute. The helper script automatically:
cities stored in `docs/cities/cities.jsonl`. The helper script automatically:

- Loads each city's latitude/longitude from the JSONL file (one JSON object per
  line).
- Places the resulting datasets under `data/<city name>/`.
- Resumes cleanly by skipping folders that already contain a `.completed`
  marker.
- Displays a `tqdm` progress bar with the total number of cities to process.

Default parameters match the request for a 2 km radius (`--radius 2000`) and
0.5 m resolution (`--resolution 0.5`). Override them along with the output
location or the number of cities to run:

```bash
# Process the first 5 cities at 3 km radius and 1 m resolution
./scripts/run_osm_overture_full_batch.sh \
  --max-cities 5 \
  --radius 3000 \
  --resolution 1.0 \
  --output-root data \
  --cities-file docs/cities/cities.jsonl
```

Re-run the script with `--force` to refresh cities that were previously marked
as completed.

**Google cost-down mode** now layers Google Place Details on top of the free OSM
and Overture downloads, then minimises paid API usage by only requesting
additional details for uncertain buildings. Tune its behaviour via environment
variables such as `GOOGLE_BUDGET_REQUESTS`, `GOOGLE_QPS_MAX`,
`UNCERTAINTY_THRESHOLD`, and `FEATURE_FILTER`; these map to the underlying
arguments of `generate_semantic_dataset_cost_down.py`.

## Comprehensive regional bundle

For full multi-provider output with additional analytics, use
`tools/multisource/generate_region_urban_bundle.py`. It produces:

1. `buildings_enriched.json` – enriched building catalogue with centroid, size,
   OSM tags, provider metadata, ratings, hours, and raw payloads for auditing.
2. `metadata.json` – summary statistics, coverage metrics, and references to all
   generated files.

Default output lives under `data/urban_region_bundle`; override with
`--output-dir`. Optional flags include `--overture-cache-dir`,
`--overture-cache-only`, and provider credential overrides. Example run:

```bash
export GOOGLE_MAPS_API_KEY="your-google-key"
python tools/multisource/generate_region_urban_bundle.py 40.7580 -73.9855 1000 \
  --output-dir data/midtown_bundle
```

## Direct Python usage

Call the main entrypoint for custom provider mixes or to attach local data:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    LAT LON RADIUS \
    --providers osm overture google local_geojson \
    --local-geojson ./path/to/data.geojson \
    --local-csv ./path/to/data.csv
```

Consult `docs/multisource/guide.md` for exhaustive CLI options, attribution
requirements, and scaling strategies.

## License

This project is released under the license in the repository root.
