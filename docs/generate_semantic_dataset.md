# Semantic Dataset Generator

This document describes the `tools/scripts/generate_semantic_dataset.py` script,
which assembles 1 m semantic rasters and multi-level building labels for a
user-defined square region.

## Overview

Given a center latitude/longitude and a half side length (in meters), the script queries
OpenStreetMap via the Overpass API to download vector features for buildings,
land-use, vegetation, water, and roads. It then:

1. Projects the features into a local metric coordinate frame centered on the
   query point.
2. Rasterizes semantic classes to a square grid whose pixel size equals the
   requested meter-per-pixel resolution (1 m by default).
3. Classifies each building polygon into primary/secondary/tertiary usage
   levels using amenity, shop, building, and land-use tags.
4. Exports the following artifacts into the chosen output directory:
   - `semantic_map.npy`: Semantic class indices encoded as an integer raster.
   - `semantic_map.png`: Grayscale preview image of the semantic raster.
   - `buildings.geojson`: Building footprints with usage hierarchy metadata.
   - `metadata.json`: Run configuration, bounding box, and summary statistics.

All operations rely on open data so the resulting dataset can be reused without
Google API calls.

## Requirements

The script depends on packages already listed in `requirement.txt`, plus
Shapely, Pillow, NumPy, and Requests:

```bash
pip install -r requirement.txt
```

If you prefer a minimal installation, the essential dependencies are:

```bash
pip install numpy requests pillow shapely
```

> **Note**: The script contacts the public Overpass API. For heavy usage you
> should self-host an Overpass instance or provide a mirrored endpoint via the
> `--overpass-url` flag.

## Usage

```bash
python tools/scripts/generate_semantic_dataset.py \
    <latitude> <longitude> <radius_meters> \
    --resolution 1.0 \
    --output /path/to/output_dir \
    --log-level INFO
```

### Positional arguments

- `<latitude>`: Decimal latitude of the region center.
- `<longitude>`: Decimal longitude of the region center.
- `<radius_meters>`: Half side length of the square AOI in meters. The raster
  covers a square of side length `2 * radius_meters` centered on the provided
  point.

### Optional arguments

- `--resolution`: Pixel size in meters (default: 1.0). Lower values increase
  raster resolution and file size.
- `--output`: Destination directory for all generated files. Defaults to
  `./semantic_dataset` relative to the current working directory.
- `--overpass-url`: Custom Overpass API endpoint. Override this when using a
  private mirror.
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.).

## Outputs

The output directory contains:

- **`semantic_map.npy`** – NumPy array of class IDs using the following index:
  - `0`: ground / background
  - `1`: vegetation
  - `2`: water
  - `3`: building
  - `4`: road (residential, tertiary, service, etc.)
  - `5`: traffic road (primary, trunk, motorway classes)
- **`semantic_map.png`** – Quick-look visualization (grayscale values follow the
  same class IDs).
- **`buildings.geojson`** – GeoJSON FeatureCollection. Each feature provides:
  - `primary_label`, `secondary_label`, `tertiary_label`
  - `source_tags`: OSM tags used to infer the hierarchy
  - Geometry footprint (WGS84), centroid coordinates, and basic attributes such
    as `height`, `building_levels`, and address fragments when available.
- **`metadata.json`** – Run metadata, bounding box, Overpass endpoint, and
  counts of fetched elements/buildings.

## Tips for Large-Scale Use

- Cache raw Overpass responses by saving `metadata.json` and `buildings.geojson`
  under region-specific names (e.g., geohash) so you can rerun rasterization
  without re-downloading vectors.
- Split large areas into overlapping squares to stay within Overpass result
  limits (~50 000 elements). The `radius` argument controls the tile size.
- Consider running a private Overpass server—or supplying a mirror via
  `--overpass-url`. The script will automatically fall back to a set of public
  mirrors when the primary endpoint returns transient errors, but a dedicated
  instance is still recommended for heavy workloads.
- Extend the classification mappings in the script to refine the building
  taxonomy for your target regions or to incorporate local zoning datasets.
