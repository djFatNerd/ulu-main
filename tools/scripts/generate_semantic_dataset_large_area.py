"""Generate semantic datasets for large areas by tiling smaller requests."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from generate_semantic_dataset import (
    CLASS_TO_ID,
    compute_bbox,
    get_numpy,
    get_pillow,
    get_requests,
    meters_per_degree_lat,
    meters_per_degree_lon,
    run_generation,
)
from semantic_palette import SEMANTIC_COLOR_PALETTE


def _tile_offsets(radius_m: float, max_tile_radius: float) -> Tuple[float, int, List[float], List[float]]:
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    if max_tile_radius <= 0:
        raise ValueError("max_tile_radius must be positive")

    max_tile_radius = min(max_tile_radius, radius_m)
    tiles_per_axis = max(1, math.ceil(radius_m / max_tile_radius))
    tile_radius = radius_m / tiles_per_axis
    step = tile_radius * 2.0
    offsets = [-radius_m + tile_radius + i * step for i in range(tiles_per_axis)]
    x_offsets = offsets
    y_offsets = list(reversed(offsets))
    return tile_radius, tiles_per_axis, x_offsets, y_offsets


def _combine_semantic_maps(
    tile_records: List[Dict[str, object]],
    tiles_per_axis: int,
    output_dir: Path,
) -> Tuple[Path, Path]:
    np = get_numpy()
    Image, _ = get_pillow()

    tile_size = None
    combined_array = None
    for record in tile_records:
        tile_map_path = Path(record["semantic_map"])  # type: ignore[arg-type]
        tile_array = np.load(tile_map_path)
        if tile_size is None:
            tile_size = tile_array.shape[0]
            combined_shape = (tiles_per_axis * tile_size, tiles_per_axis * tile_size)
            combined_array = np.zeros(combined_shape, dtype=tile_array.dtype)
        row = int(record["row"])  # type: ignore[index]
        col = int(record["col"])  # type: ignore[index]
        combined_array[
            row * tile_size : (row + 1) * tile_size,
            col * tile_size : (col + 1) * tile_size,
        ] = tile_array

    if combined_array is None or tile_size is None:
        raise RuntimeError("No semantic maps were generated for combination")

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_map_path = combined_dir / "semantic_map.npy"
    np.save(combined_map_path, combined_array)

    grayscale_path = combined_map_path.with_suffix(".png")
    Image.fromarray(combined_array, mode="L").save(grayscale_path)

    max_class_id = max(CLASS_TO_ID.values())
    palette = np.zeros((max_class_id + 1, 3), dtype=np.uint8)
    for class_name, class_id in CLASS_TO_ID.items():
        palette[class_id] = SEMANTIC_COLOR_PALETTE[class_name]
    colored = palette[combined_array]
    color_path = combined_dir / "semantic_map_colored.png"
    Image.fromarray(colored, mode="RGB").save(color_path)

    return combined_map_path, color_path


def _merge_building_geojson(
    tile_records: List[Dict[str, object]], output_dir: Path
) -> Tuple[Path, int]:
    features: Dict[str, Dict] = {}
    for record in tile_records:
        geojson_path = Path(record["buildings_geojson"])  # type: ignore[arg-type]
        if not geojson_path.exists():
            continue
        with geojson_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, feature in enumerate(data.get("features", [])):
            feature_id = feature.get("id")
            if feature_id is None:
                feature_id = f"{record['row']}_{record['col']}_{idx}"
            features[str(feature_id)] = feature

    collection = {
        "type": "FeatureCollection",
        "features": list(features.values()),
    }

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = combined_dir / "buildings.geojson"
    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)
    return geojson_path, len(collection["features"])


def generate_large_area(
    lat: float,
    lon: float,
    radius_m: float,
    resolution: float,
    output_dir: Path,
    overpass_url: str,
    max_tile_radius: float,
) -> None:
    tile_radius, tiles_per_axis, x_offsets, y_offsets = _tile_offsets(radius_m, max_tile_radius)

    meters_lat = meters_per_degree_lat(lat)
    meters_lon = meters_per_degree_lon(lat)

    requests_module = get_requests()
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_root = output_dir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)

    total_tiles = tiles_per_axis * tiles_per_axis
    tile_records: List[Dict[str, object]] = []
    logging.info(
        "Processing %d tiles (tile radius %.2f m) to cover %.2f m query radius",
        total_tiles,
        tile_radius,
        radius_m,
    )

    with tqdm(total=total_tiles, desc="Tiles", unit="tile") as progress:
        for row_idx, y_offset in enumerate(y_offsets):
            tile_lat = lat + (y_offset / meters_lat)
            for col_idx, x_offset in enumerate(x_offsets):
                tile_lon = lon + (x_offset / meters_lon)
                tile_output = tiles_root / f"tile_r{row_idx:02d}_c{col_idx:02d}"
                tile_output.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "Tile r%02d c%02d centered at (%.6f, %.6f) with radius %.2f m",
                    row_idx,
                    col_idx,
                    tile_lat,
                    tile_lon,
                    tile_radius,
                )
                run_generation(
                    lat=tile_lat,
                    lon=tile_lon,
                    radius_m=tile_radius,
                    resolution=resolution,
                    output_dir=tile_output,
                    overpass_url=overpass_url,
                    requests_module=requests_module,
                )

                metadata_path = tile_output / "metadata.json"
                buildings_geojson = tile_output / "buildings.geojson"
                semantic_map = tile_output / "semantic_map.npy"

                tile_info = {
                    "row": row_idx,
                    "col": col_idx,
                    "center": {"lat": tile_lat, "lon": tile_lon},
                    "radius_m": tile_radius,
                    "output_dir": str(tile_output),
                    "metadata_path": str(metadata_path),
                    "buildings_geojson": str(buildings_geojson),
                    "semantic_map": str(semantic_map),
                }

                if metadata_path.exists():
                    with metadata_path.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    tile_info["metadata"] = metadata

                tile_records.append(tile_info)
                progress.update(1)

    combined_map_path, color_path = _combine_semantic_maps(
        tile_records, tiles_per_axis, output_dir
    )
    combined_geojson_path, total_buildings = _merge_building_geojson(
        tile_records, output_dir
    )

    bbox = compute_bbox(lat, lon, radius_m)
    total_elements = 0
    for record in tile_records:
        tile_metadata = record.get("metadata")
        if isinstance(tile_metadata, dict):
            total_elements += tile_metadata.get("element_count", 0)
    metadata_tiles: List[Dict[str, object]] = []
    for record in tile_records:
        tile_metadata = record.get("metadata")
        if not isinstance(tile_metadata, dict):
            tile_metadata = {}
        metadata_tiles.append(
            {
                "row": record["row"],
                "col": record["col"],
                "center": record["center"],
                "radius_m": record["radius_m"],
                "output_dir": record["output_dir"],
                "metadata_path": record["metadata_path"],
                "semantic_map": record["semantic_map"],
                "buildings_geojson": record["buildings_geojson"],
                "element_count": tile_metadata.get("element_count"),
                "building_count": tile_metadata.get("building_count"),
            }
        )

    combined_metadata = {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "resolution_m": resolution,
        "bbox": {
            "south": bbox[0],
            "west": bbox[1],
            "north": bbox[2],
            "east": bbox[3],
        },
        "tile_radius_m": tile_radius,
        "tiles_per_axis": tiles_per_axis,
        "tile_count": len(tile_records),
        "class_to_id": CLASS_TO_ID,
        "semantic_map": {
            "npy": str(combined_map_path),
            "png": str(combined_map_path.with_suffix(".png")),
            "colored_png": str(color_path),
        },
        "buildings_geojson": str(combined_geojson_path),
        "tiles": metadata_tiles,
        "overpass_url": overpass_url,
        "element_count": total_elements,
        "building_count": total_buildings,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, ensure_ascii=False, indent=2)
    logging.info("Large area generation complete: %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic maps for a large area by tiling requests to the "
            "standard dataset generator."
        )
    )
    parser.add_argument("lat", type=float, help="Center latitude in decimal degrees")
    parser.add_argument("lon", type=float, help="Center longitude in decimal degrees")
    parser.add_argument(
        "radius",
        type=float,
        help="Half side length of the total square region in meters",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Raster resolution in meters per pixel (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./semantic_dataset_large_area"),
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--overpass-url",
        type=str,
        default="https://overpass-api.de/api/interpreter",
        help="Overpass API endpoint",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=1500.0,
        help="Maximum tile radius in meters used to split the large query",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    generate_large_area(
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius,
        resolution=args.resolution,
        output_dir=args.output,
        overpass_url=args.overpass_url,
        max_tile_radius=args.max_radius,
    )


if __name__ == "__main__":
    main()
