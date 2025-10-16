# Semantic Map Dataset Toolkit

This repository packages the utilities we use to build semantic map tiles and
building-level metadata from several data providers. The core Python entrypoint
(`tools/multisource/generate_semantic_dataset_enriched.py`) accepts a latitude,
longitude, search radius, and raster resolution, then enriches each OpenStreetMap
(OSM) footprint with optional third-party attributes.

## Installation

Install the Python dependencies once before running any of the scripts:

```bash
pip install -r requirements.txt
```

## Resources

We currently rely on four categories of resources. The scripted combinations
below are built on top of these components. Each subsection explains how to
obtain credentials (when needed), typical costs, and operational tips.

### A. OpenStreetMap (OSM)

- **What it provides** – public building footprints, roads, and semantic tags.
- **How we access it** – all scripts call the Overpass API (default endpoint
  `https://overpass-api.de/api/interpreter`). You may override the endpoint via
  `OVERPASS_URL` or define a `FALLBACK_OVERPASS` URL to improve reliability.
- **Credential requirements** – none. Overpass is public, but heavy usage
  should respect its fair-use policy. For large batch jobs consider hosting your
  own Overpass instance or staggering requests with `REQUEST_SLEEP`.

### B. Overture Maps downloads

- **What it provides** – global, community-maintained place and building
  attributes (names, categories, brand metadata) from the Overture Maps
  Foundation.
- **How we access it** – install the
  [`overturemaps` Python package](https://pypi.org/project/overturemaps/) and use
  its bundled CLI to download GeoJSON for the query bounding box. The scripts in
  this repository invoke `python -m overturemaps.cli download ...` under the
  hood, so make sure the package is available in the active environment. You can
  test connectivity manually:

  ```bash
  pip install overturemaps
  overturemaps download --bbox=-71.068,42.353,-71.058,42.363 -f geojson --type=building -o sample.geojson
  ```
- **Credential requirements** – none. Data is fetched from the public Overture
  dataset releases; no API token is needed.
- **Cost and usage notes** – downloads are free but respect the data licence and
  avoid hammering the service with very fine-grained bounding boxes.

### C. Google Places API

- **What it provides** – commercial point-of-interest metadata including
  official names, place types, ratings, hours, phone numbers, and website links.
- **How to obtain an API key** –
  1. Sign in to the [Google Cloud Console](https://console.cloud.google.com/)
     and create (or reuse) a project.
  2. Enable the *Places API* and *Maps Places API* services for that project.
  3. Navigate to **APIs & Services → Credentials**, create an API key, and
     restrict it to the Places APIs plus your desired HTTP referrers or IPs.
  4. Set the key as `GOOGLE_MAPS_API_KEY` (or `GOOGLE_API_KEY`) before running a
     script.
- **Cost and usage notes** – Google requires an active billing account even for
  evaluation. Each account receives USD $200 of free usage credit per month; API
  calls beyond that credit are billed per request according to the Places API
  rate card. Monitor quotas in the Cloud Console and consider setting daily
  limits or alerts.

### D. Local offline data

- **What it provides** – any custom GeoJSON or CSV datasets that you maintain
  (e.g., municipal inventories or proprietary sources).
- **How we access it** – invoke
  `tools/multisource/generate_semantic_dataset_enriched.py` directly and supply
  the `--local-geojson` and/or `--local-csv` flags.
- **Cost and usage notes** – managed entirely by you. Be sure to follow the
  licensing terms of any third-party datasets you ingest.

## Combination scripts

Each script under `scripts/` accepts the same positional parameters:

```
./scripts/<script_name>.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
```

- `LAT` / `LON` – decimal degrees identifying the query center.
- `RADIUS` – half the side length of the square (meters).
- `RESOLUTION` – optional raster resolution in meters (defaults to `1.0`).
- `OUTPUT_DIR` – optional destination directory (defaults per combination).

The scripts also honour advanced environment overrides such as `MATCH_DISTANCE`,
`PROVIDER_RADIUS`, `REQUEST_SLEEP`, `LOG_LEVEL`, `OVERPASS_URL`, and
`FALLBACK_OVERPASS` for OSM, plus the provider-specific variables described in
each section.

| Combination | Script | Providers used | Required credentials |
| --- | --- | --- | --- |
| 1. OSM only | `run_osm.sh` | OSM labels only | none |
| 2. Overture only | `run_overture.sh` | Overture downloads | `overturemaps` package |
| 3. Google only | `run_google.sh` | Google Places | `GOOGLE_MAPS_API_KEY` |
| 4. OSM + Google | `run_osm_google.sh` | OSM labels merged with Google Places | `GOOGLE_MAPS_API_KEY` |
| 5. OSM + Overture | `run_osm_overture.sh` | OSM labels merged with Overture downloads | `overturemaps` package |
| 6. Overture + Google | `run_overture_google.sh` | Overture downloads + Google Places (no OSM labels) | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |
| 7. OSM + Overture + Google | `run_osm_overture_google.sh` | OSM labels merged with Overture and Google Places | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |

### Combination 1 – OSM only

```
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm
```

Only Overpass-sourced attributes are preserved. No external credentials are
required.

### Combination 2 – Overture only

```
./scripts/run_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture
```

This workflow replaces OSM semantic labels with attributes from the Overture
downloads handled via the `overturemaps` CLI. Optional environment overrides:

- `OVERTURE_THEME` – dataset theme (default `buildings`).
- `OVERTURE_LIMIT` – per-request feature cap.
- `OVERTURE_INCLUDE_FIELDS`, `OVERTURE_CATEGORY_FIELDS`, `OVERTURE_NAME_FIELDS`
  – space-delimited field overrides for property extraction.
- `OVERTURE_TIMEOUT` – CLI execution timeout in seconds.
- `OVERTURE_CACHE_DIR` – directory used to persist Overture responses for reuse (default `data/overture_cache`).
- `--overture-cache-only` / `OVERTURE_CACHE_ONLY=1` – disable fresh downloads and rely solely on cached payloads.
- `OVERTURE_CACHE_DIR` – directory used to persist Overture responses for reuse (default `data/overture_cache`).
- `--overture-cache-only` / `OVERTURE_CACHE_ONLY=1` – disable fresh downloads and rely solely on files already present in the
  cache directory. Populate the cache by running the workflow once while online or by copying pre-downloaded payloads into the
  directory using the naming scheme produced by the online runs.

### Combination 3 – Google only

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_google
```

The dataset retains OSM geometry but derives semantic labels purely from Google
Places. Adjust `REQUEST_SLEEP` to manage quota.

### Combination 4 – OSM + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_google
```

Combines native OSM categories with Google Places metadata, producing a blended
hierarchy and additional provider fields (ratings, hours, etc.).

### Combination 5 – OSM + Overture

```
./scripts/run_osm_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture
```

Keeps OSM labels while enriching features with Overture-provided names and
categories. Supports the same Overture environment overrides listed in
Combination 2.

### Combination 6 – Overture + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture_google
```

Outputs a provider-only dataset that fuses Overture and Google attributes while
suppressing the original OSM semantic labels.

### Combination 7 – OSM + Overture + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google
```

## Resources

We currently rely on four categories of resources. The scripted combinations
below are built on top of these components.

- **A. OpenStreetMap (OSM)** – public geometry and semantic labels accessed via
  the Overpass API. This is always the geometric foundation of the dataset.
- **B. Overture Maps downloads** – global building/business attributes donated
  by the Overture community (Meta, Microsoft, Amazon, Esri, etc.) retrieved via
  the `overturemaps` CLI.
- **C. Google Places API** – commercial point-of-interest metadata (official
  names, categories, ratings, and opening hours). Requires a
  `GOOGLE_MAPS_API_KEY` (or `GOOGLE_API_KEY`).
- **D. Local offline data** – optional GeoJSON or CSV files that you maintain.
  These can be wired in manually with the Python entrypoint using
  `--local-geojson`/`--local-csv` when you need to augment the scripted
  combinations.

## Combination scripts

Each script under `scripts/` accepts the same positional parameters:

```
./scripts/<script_name>.sh LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
```

- `LAT` / `LON` – decimal degrees identifying the query center.
- `RADIUS` – half the side length of the square (meters).
- `RESOLUTION` – optional raster resolution in meters (defaults to `1.0`).
- `OUTPUT_DIR` – optional destination directory (defaults per combination).

The scripts also honour advanced environment overrides such as `MATCH_DISTANCE`,
`PROVIDER_RADIUS`, `REQUEST_SLEEP`, `LOG_LEVEL`, `OVERPASS_URL`, and
`FALLBACK_OVERPASS` for OSM, plus the provider-specific variables described in
each section.

| Combination | Script | Providers used | Required credentials |
| --- | --- | --- | --- |
| 1. OSM only | `run_osm.sh` | OSM labels only | none |
| 2. Overture only | `run_overture.sh` | Overture downloads | `overturemaps` package |
| 3. Google only | `run_google.sh` | Google Places | `GOOGLE_MAPS_API_KEY` |
| 4. OSM + Google | `run_osm_google.sh` | OSM labels merged with Google Places | `GOOGLE_MAPS_API_KEY` |
| 5. OSM + Overture | `run_osm_overture.sh` | OSM labels merged with Overture downloads | `overturemaps` package |
| 6. Overture + Google | `run_overture_google.sh` | Overture downloads + Google Places (no OSM labels) | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |
| 7. OSM + Overture + Google | `run_osm_overture_google.sh` | OSM labels merged with Overture and Google Places | `overturemaps` package, `GOOGLE_MAPS_API_KEY` |

### Combination 1 – OSM only

```
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm
```

Only Overpass-sourced attributes are preserved. No external credentials are
required.

### Combination 2 – Overture only

```
./scripts/run_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture
```

This workflow replaces OSM semantic labels with attributes from the Overture
downloads handled via the `overturemaps` CLI. Optional environment overrides:

- `OVERTURE_THEME` – dataset theme (default `buildings`).
- `OVERTURE_LIMIT` – per-request feature cap.
- `OVERTURE_INCLUDE_FIELDS`, `OVERTURE_CATEGORY_FIELDS`, `OVERTURE_NAME_FIELDS`
  – space-delimited field overrides for property extraction.
- `OVERTURE_TIMEOUT` – CLI execution timeout in seconds.

### Combination 3 – Google only

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_google
```

The dataset retains OSM geometry but derives semantic labels purely from Google
Places. Adjust `REQUEST_SLEEP` to manage quota.

### Combination 4 – OSM + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_google
```

Combines native OSM categories with Google Places metadata, producing a blended
hierarchy and additional provider fields (ratings, hours, etc.).

### Combination 5 – OSM + Overture

```
./scripts/run_osm_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture
```

Keeps OSM labels while enriching features with Overture-provided names and
categories. Supports the same Overture environment overrides listed in
Combination 2.

### Combination 6 – Overture + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture_google
```

Outputs a provider-only dataset that fuses Overture and Google attributes while
suppressing the original OSM semantic labels.

### Combination 7 – OSM + Overture + Google

```
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google
```

Uses OSM geometry and labels as the backbone while layering both provider APIs
on top. Ideal when you want the fullest set of enrichment fields in a single
pass.

## Extending with local data (Resource D)

When you need to mix in municipal data portals or private registries, call the
Python entrypoint directly and supply your files:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    LAT LON RADIUS \
    --providers local_geojson local_csv \
    --local-geojson ./path/to/data.geojson \
    --local-csv ./path/to/data.csv
```

Refer to `docs/multisource/guide.md` for detailed CLI flags, attribution
requirements, and scaling guidance.

## License

This project is released under the license included in the repository root.
