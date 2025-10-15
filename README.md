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

### B. Overture Maps Places API

- **What it provides** – global, community-maintained place and building
  attributes (names, categories, brand metadata) from the Overture Maps
  Foundation.
- **How to obtain an access token** –
  1. Visit the [Overture Maps developer portal](https://developer.overturemaps.org/).
  2. Sign in with a GitHub account and create a new application.
  3. Generate an API token and copy it to a secure location.
  4. Export it as `OVERTURE_AUTH_TOKEN` before running the scripts.
- **Cost and usage notes** – Overture currently offers free access with rate
  limits while the API is in preview. Monitor the portal for production pricing
  changes. Rotate tokens if they are compromised and avoid committing them to
  version control.

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
| 2. Overture only | `run_overture.sh` | Overture Places | `OVERTURE_AUTH_TOKEN` |
| 3. Google only | `run_google.sh` | Google Places | `GOOGLE_MAPS_API_KEY` |
| 4. OSM + Google | `run_osm_google.sh` | OSM labels merged with Google Places | `GOOGLE_MAPS_API_KEY` |
| 5. OSM + Overture | `run_osm_overture.sh` | OSM labels merged with Overture Places | `OVERTURE_AUTH_TOKEN` |
| 6. Overture + Google | `run_overture_google.sh` | Overture Places + Google Places (no OSM labels) | `OVERTURE_AUTH_TOKEN`, `GOOGLE_MAPS_API_KEY` |
| 7. OSM + Overture + Google | `run_osm_overture_google.sh` | OSM labels merged with Overture and Google Places | `OVERTURE_AUTH_TOKEN`, `GOOGLE_MAPS_API_KEY` |

### Combination 1 – OSM only

```
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm
```

Only Overpass-sourced attributes are preserved. No external credentials are
required.

### Combination 2 – Overture only

```
OVERTURE_AUTH_TOKEN=... ./scripts/run_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture
```

This workflow replaces OSM semantic labels with attributes from the Overture
Maps Places API. Optional environment overrides:

- `OVERTURE_ENDPOINT` – alternate API endpoint.
- `OVERTURE_THEME` – Overture theme (default `buildings`).
- `OVERTURE_LIMIT` – per-request feature cap.
- `OVERTURE_INCLUDE_FIELDS`, `OVERTURE_CATEGORY_FIELDS`, `OVERTURE_NAME_FIELDS`
  – space-delimited field overrides.
- `OVERTURE_TIMEOUT` – HTTP timeout in seconds.

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
OVERTURE_AUTH_TOKEN=... ./scripts/run_osm_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture
```

Keeps OSM labels while enriching features with Overture-provided names and
categories. Supports the same Overture environment overrides listed in
Combination 2.

### Combination 6 – Overture + Google

```
OVERTURE_AUTH_TOKEN=... \
GOOGLE_MAPS_API_KEY=... ./scripts/run_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture_google
```

Outputs a provider-only dataset that fuses Overture and Google attributes while
suppressing the original OSM semantic labels.

### Combination 7 – OSM + Overture + Google

```

## Resources

We currently rely on four categories of resources. The scripted combinations
below are built on top of these components.

- **A. OpenStreetMap (OSM)** – public geometry and semantic labels accessed via
  the Overpass API. This is always the geometric foundation of the dataset.
- **B. Overture Maps Places API** – global building/business attributes donated
  by the Overture community (Meta, Microsoft, Amazon, Esri, etc.). Requires an
  API token via `OVERTURE_AUTH_TOKEN`.
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
| 2. Overture only | `run_overture.sh` | Overture Places | `OVERTURE_AUTH_TOKEN` |
| 3. Google only | `run_google.sh` | Google Places | `GOOGLE_MAPS_API_KEY` |
| 4. OSM + Google | `run_osm_google.sh` | OSM labels merged with Google Places | `GOOGLE_MAPS_API_KEY` |
| 5. OSM + Overture | `run_osm_overture.sh` | OSM labels merged with Overture Places | `OVERTURE_AUTH_TOKEN` |
| 6. Overture + Google | `run_overture_google.sh` | Overture Places + Google Places (no OSM labels) | `OVERTURE_AUTH_TOKEN`, `GOOGLE_MAPS_API_KEY` |
| 7. OSM + Overture + Google | `run_osm_overture_google.sh` | OSM labels merged with Overture and Google Places | `OVERTURE_AUTH_TOKEN`, `GOOGLE_MAPS_API_KEY` |

### Combination 1 – OSM only

```
./scripts/run_osm.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm
```

Only Overpass-sourced attributes are preserved. No external credentials are
required.

### Combination 2 – Overture only

```
OVERTURE_AUTH_TOKEN=... ./scripts/run_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture
```

This workflow replaces OSM semantic labels with attributes from the Overture
Maps Places API. Optional environment overrides:

- `OVERTURE_ENDPOINT` – alternate API endpoint.
- `OVERTURE_THEME` – Overture theme (default `buildings`).
- `OVERTURE_LIMIT` – per-request feature cap.
- `OVERTURE_INCLUDE_FIELDS`, `OVERTURE_CATEGORY_FIELDS`, `OVERTURE_NAME_FIELDS`
  – space-delimited field overrides.
- `OVERTURE_TIMEOUT` – HTTP timeout in seconds.

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
OVERTURE_AUTH_TOKEN=... ./scripts/run_osm_overture.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture
```

Keeps OSM labels while enriching features with Overture-provided names and
categories. Supports the same Overture environment overrides listed in
Combination 2.

### Combination 6 – Overture + Google

```
OVERTURE_AUTH_TOKEN=... \
GOOGLE_MAPS_API_KEY=... ./scripts/run_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_overture_google
```
OVERTURE_AUTH_TOKEN=... \
GOOGLE_MAPS_API_KEY=... ./scripts/run_osm_overture_google.sh 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google
```

Uses OSM geometry and labels as the backbone while layering both provider APIs
on top. Ideal when you want the fullest set of enrichment fields in a single
pass.

Outputs a provider-only dataset that fuses Overture and Google attributes while
suppressing the original OSM semantic labels.

### Combination 7 – OSM + Overture + Google

```
OVERTURE_AUTH_TOKEN=... \
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
