# Large-Area Semantic Dataset Generator

The `tools/osm/generate_semantic_dataset_large_area.py` helper script wraps
`tools/osm/generate_semantic_dataset.py` to support very large regions. It does **not**
run automatically; you should choose it when the requested area risks exceeding
Overpass limits or local memory. The helper automatically splits the query
square into smaller tiles, renders each tile sequentially, and then stitches the
semantic rasters and building GeoJSON back together.
## Why use the tiling helper?

- **Stay within Overpass limits.** Very large bounding boxes can exceed the
  public API's element limits or time out. Tiling keeps each request compact.
- **Bounded memory footprint.** Rasterization happens per tile, so peak memory
  and disk usage match the tile size rather than the full region.
- **Deterministic coverage.** Tiles are laid out without gaps so features along
  the tile borders are preserved.
- **Progress tracking.** A `tqdm` progress indicator reports how many of the
  `N` tiles have finished (`1/N`, `2/N`, …) as the script runs.

## Usage

```bash
python -m tools.osm.generate_semantic_dataset_large_area \
    <latitude> <longitude> <radius_meters> \
    --max-radius 1500 \
    --resolution 1.0 \
    --output /path/to/output_dir \
    --log-level INFO
```

> **Dependency**: Install `tqdm` (`pip install tqdm`) to enable the progress
> indicator used by the tiling helper.

### Positional arguments

- `<latitude>` / `<longitude>` – Center of the overall square region.
- `<radius_meters>` – Half side length of the overall square in meters. The
  script will cover a square with side `2 * radius_meters` centered on the
  provided coordinates.

### Key options

- `--max-radius` – Maximum tile radius (meters). The helper computes a grid so
  that every tile radius is `<= max-radius`, ensuring all tiles fit within the
  Overpass limits you choose.
- `--resolution` – Raster resolution passed through to each tile (meters per
  pixel). Defaults to 1 m.
- `--output` – Root directory for results. The script creates per-tile folders
  under `tiles/` and writes merged artifacts to `combined/`.
- `--overpass-url` – Optional Overpass API mirror.
- `--log-level` – Standard logging verbosity.

## Outputs

The output directory includes both per-tile and merged artifacts:

- `tiles/tile_rXX_cYY/…` – Each tile contains the same outputs produced by
  `tools/osm/generate_semantic_dataset.py` (semantic rasters, colored previews, building
  metadata, and tile-level metadata JSON).
- `combined/semantic_map.npy` – Full-resolution semantic raster assembled from
  all tiles. The helper also writes `semantic_map.png` (grayscale) and
  `semantic_map_colored.png` (RGB) previews using the shared palette.
- `combined/buildings.geojson` – Merged FeatureCollection of buildings across
  every tile. Duplicate features that span multiple tiles are deduplicated by
  OSM ID.
- `metadata.json` – Summary of the entire run, including the global bounding
  box, tile grid configuration, cumulative element/building counts, and
  references to every per-tile artifact.

The combined metadata and semantic maps ensure you have both the granular tile
outputs and an aggregated view suitable for downstream processing.
