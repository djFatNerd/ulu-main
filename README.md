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

### Available combinations

| Script | Providers merged | Credentials required |
| --- | --- | --- |
| `run_osm.sh` | OSM only | none |
| `run_overture.sh` | Overture only | `overturemaps` package |
| `run_google.sh` | Google Places on top of OSM geometry | `GOOGLE_MAPS_API_KEY` |
| `run_osm_google.sh` | OSM + Google Places | `GOOGLE_MAPS_API_KEY` |
| `run_osm_overture.sh` | OSM + Overture Maps | `overturemaps` package |
| `run_overture_google.sh` | Overture Maps + Google Places | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |
| `run_osm_overture_google.sh` | OSM + Overture Maps + Google Places | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |

Example invocations:

```bash
# OSM-only dataset
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm

# OSM + Google Places enrichment
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_google

# Overture-only dataset
./scripts/run_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture
```

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
