# Multisource Semantic Dataset Workflow

The multisource workflow augments the existing OpenStreetMap (OSM) rasterization
pipeline with external points-of-interest providers. It produces enriched
GeoJSON metadata that combines OSM building footprints with provider attributes
such as official names, category hierarchies, ratings, and opening hours.

## Components

- `tools/multisource/generate_semantic_dataset_enriched.py` orchestrates the
  pipeline. It reuses the `tools.osm` rasterization utilities to generate the
  semantic raster and building features, then performs centroid-based lookups
  against configured providers.
- Providers are pluggable. The initial implementation includes:
  - **OSM only** (`--provider osm`): preserves the original OSM labels without
    external enrichment.
  - **Google Places only** (`--provider google`): uses OSM footprints for
    geometry but derives categories entirely from Google types.
  - **Hybrid OSM + Google** (`--provider osm_google`, default): merges OSM
    building classifications with Google types to build a combined
    primary-secondary-tertiary category path.

## Dependencies and configuration

Install the standard dependencies plus the Google Places client:

```bash
pip install -r requirements.txt
```

Set the Google Maps Platform key via an environment variable or CLI flag:

```bash
export GOOGLE_MAPS_API_KEY="<your-api-key>"
# or pass --google-api-key directly when invoking the script
```

Example invocation that merges OSM and Google metadata:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --resolution 1.0 \
    --output ./times_square_enriched \
    --provider osm_google \
    --request-sleep 0.5 \
    --match-distance 30
```

To opt into Google-only classifications (retaining OSM footprints but not the
OSM semantic labels):

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --provider google
```

If you need to fall back to an OSM-only workflow while keeping the enriched
schema for downstream compatibility:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --provider osm
```

## Output schema additions

The enriched GeoJSON augments each building feature with:

- `enriched_primary_label`, `enriched_secondary_label`,
  `enriched_tertiary_label`, and `enriched_category_path` describing the merged
  category hierarchy. Provenance is tracked via `enriched_category_provenance`
  (`osm_only`, `provider_only`, or `osm+provider`).
- Provider metadata (`enriched_name`, `enriched_rating`,
  `enriched_rating_count`, `enriched_opening_hours`) with per-field provenance
  flags so downstream systems know whether a value came from Google, OSM, or is
  absent.
- Diagnostic attributes such as `provider_place_id`, `provider_distance_m`, and
  `provider_confidence` to expose the spatial matching status.

`metadata.json` includes an `enrichment` block summarizing the provider mode,
match rate, fields added, and whether OSM labels were used during category
composition.

## Rate limiting and quota management

External providers enforce strict quotas. The CLI exposes `--request-sleep`
(default `0.2` seconds) to throttle requests and avoid 429 errors. Increase the
sleep duration for large areas or when using shared API keys. The Google Places
API also limits the number of nearby searches per day; consult your quota in the
Google Cloud console and adjust the radius (`--provider-radius`) or scope of
your queries accordingly.

## Privacy and responsible usage

- Do not store or redistribute personally identifiable information (PII). The
  enrichment step only records business-facing metadata surfaced by the provider
  APIs.
- Respect the **Google Maps Platform Terms of Service** and **OpenStreetMap
  usage policy**. Some jurisdictions restrict combining datasets; review local
  regulations before deployment.
- Cache provider responses responsibly. If you persist raw provider payloads
  (`provider_raw`), secure them and avoid sharing beyond teams covered by the
  provider agreements.

## Attribution requirements

- Retain the standard OpenStreetMap attribution wherever the geometry or OSM
  tags are displayed or redistributed.
- Follow Google Maps attribution guidelines when any Google-derived fields
  (names, ratings, opening hours, etc.) are displayed. Include "Data © Google"
  or the attribution string required for your contract.
- Document downstream usage so that consumers of the enriched dataset know that
  third-party terms apply to the provider-specific fields.

## Extending to new providers

The provider layer is pluggable—implement a subclass of `ProviderBase` that
returns a `ProviderResult`. You can use the `NullProvider` as a template for
providers that rely exclusively on OSM or open datasets. When adding new
providers, update this guide with rate-limit guidance and attribution rules.

### Additional open-data providers

Beyond the bundled Google Places integration, the CLI now ships with
providers that can operate entirely on offline or open datasets. The
`--provider` shortcut flag offers a few presets:

| `--provider` value | Resulting behaviour |
| --- | --- |
| `osm` | Only OpenStreetMap labels are kept; no external enrichment runs. |
| `google` | Only Google Places data is requested (requires an API key). |
| `osm_google` *(default)* | OSM labels combined with Google Places metadata when a key is supplied. |
| `overture` | Only the Overture Maps dataset is downloaded for each building via the `overturemaps` CLI. |
| `osm_overture` | OSM labels combined with the Overture Maps downloads. |

You can supply `--providers` for more complex mixes—for example
`--providers overture local_geojson`—but the shortcuts above are often
enough for quick runs.

- `overture` downloads the [Overture Maps](https://docs.overturemaps.org/)
  building dataset using the `overturemaps` Python package. Install it with
  `pip install overturemaps` so the bundled CLI is available. Set
  `--provider overture` to rely exclusively on Overture metadata, or combine it
  with OSM via `--provider osm_overture` or the generic
  `--providers overture ...` flag. You can customize the query radius and
  property extraction through the `--overture-*` arguments:

  ```bash
  python tools/multisource/generate_semantic_dataset_enriched.py \
      48.8566 2.3522 1500 \
      --provider osm_overture \
      --overture-include-fields names categories addresses \
      --overture-category-fields categories function building.use \
      --overture-name-fields name names.primary \
      --request-sleep 0.1
  ```

  The CLI currently accesses public dataset releases and does not require
  authentication. Use `--overture-timeout` to adjust tolerance for long-running
  downloads. The provider falls back to OSM labels when Overture does not report
  categories.

  To run fully offline, pre-populate the cache directory (default
  `data/overture_cache`) with responses gathered during an online session, then
  re-run the workflow using `--overture-cache-only` or setting
  `OVERTURE_CACHE_ONLY=1`. In cache-only mode the script never contacts the API;
  if a bounding box is missing, the corresponding buildings will simply be left
  without Overture enrichment.

- `local_geojson` ingests a FeatureCollection from disk (for example, Microsoft
  building footprints, government POI catalogs, or Mapillary exports). Configure
  the relevant property fields via `--local-geojson-*` flags so that names,
  categories, ratings, and opening hours align with the enriched schema.
- `local_csv` targets lightweight tabular sources such as business registries or
  cached provider responses. Supply column names through
  `--local-csv-*` arguments and the loader will build a spatial index for
  centroid matching.

Both providers respect the shared `--match-distance` tolerance and expose
per-field provenance so that downstream systems can trace exactly which source
contributed each attribute.

### Multi-source orchestration

Use the new `--providers` flag to combine multiple sources in a single pass.
For example, the following merges OSM labels with a local GeoJSON catalog and a
CSV business registry while still falling back to Google for gaps:

```bash
python tools/multisource/generate_semantic_dataset_enriched.py \
    40.7580 -73.9855 1000 \
    --providers google local_geojson local_csv \
    --local-geojson ./data/government_poi.geojson \
    --local-geojson-name-field official_name \
    --local-geojson-category-fields sector division \
    --local-csv ./data/business_registry.csv \
    --local-csv-name-field trade_name \
    --local-csv-category-fields naics_description \
    --output ./times_square_multi
```

The lookup pipeline runs each provider sequentially, merging categories,
names, and ratings while tracking field-level provenance. Disable the
OSM taxonomy contribution (for Google-only or non-OSM datasets) with
`--disable-osm-labels`.

### Turnkey multi-resource script

For reproducible runs, invoke `./scripts/multi-resource.sh`. The helper accepts
the standard positional arguments (`LAT LON RADIUS [OUTPUT_DIR]`) and exposes
per-provider toggles via environment variables:

```bash
ENABLE_GOOGLE=true \
LOCAL_GEOJSON_PATH=./data/city_open_data.geojson \
LOCAL_CSV_PATH=./data/licensed_businesses.csv \
./scripts/multi-resource.sh 40.7580 -73.9855 1000 ./times_square_multi
```

Common overrides include `DISABLE_OSM_LABELS=true`,
`LOCAL_GEOJSON_CATEGORY_FIELDS="primary secondary"`, and
`LOCAL_CSV_CATEGORY_FIELDS="industry_subclass"`. The script validates file
paths, applies shared throttling parameters, and forwards additional CLI flags
to the Python entry point so teams can compose cost-effective, high-confidence
datasets without editing code.

### Free-tier mode

When you need a fully cost-free run, pass `--free-tier` (or export
`FREE_TIER=true` when using the helper script). This disables paid providers,
keeps the enriched OSM schema, and automatically loads the sample open datasets
bundled under `data/open/`. Override them via `--free-tier-geojson` or
`--free-tier-csv` to point at Microsoft building footprints, Mapillary POIs, or
other municipal open data. The script gracefully falls back to OSM-only labels
if no offline datasets are available.
