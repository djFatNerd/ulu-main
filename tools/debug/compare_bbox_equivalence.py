#!/usr/bin/env python3
"""Compare Overture downloads between a single bbox and tiled requests."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import shape

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.multisource.generate_semantic_dataset_enriched import (  # noqa: E402
    OvertureBuildingsProvider,
    _deduplicate_features,
    _stable_feature_id,
)
from tools.multisource.generate_semantic_dataset_osm_overture_full import (  # noqa: E402
    DEFAULT_OVERTURE_INCLUDE_FIELDS,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORY_FIELDS = [
    "categories",
    "category",
    "function",
    "building.use",
    "building.function",
]
DEFAULT_NAME_FIELDS = [
    "name",
    "names.primary",
    "display.name",
    "displayName.text",
]


def parse_tile_grid(raw: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", raw)
    if not match:
        raise argparse.ArgumentTypeError("--tile must be of the form NxM (e.g. 2x2)")
    nx, ny = int(match.group(1)), int(match.group(2))
    if nx <= 0 or ny <= 0:
        raise argparse.ArgumentTypeError("Tile dimensions must be positive integers")
    return nx, ny


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbox",
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        type=float,
        required=True,
        help="Bounding box to compare (west south east north)",
    )
    parser.add_argument("--overture-endpoint", default="https://api.overturemaps.org/places/v1/places")
    parser.add_argument("--overture-theme", default="buildings")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--include-fields",
        nargs="*",
        default=DEFAULT_OVERTURE_INCLUDE_FIELDS,
    )
    parser.add_argument(
        "--category-fields",
        nargs="*",
        default=DEFAULT_CATEGORY_FIELDS,
    )
    parser.add_argument(
        "--name-fields",
        nargs="*",
        default=DEFAULT_NAME_FIELDS,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./overture_cache_debug"),
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--tile",
        type=parse_tile_grid,
        default=parse_tile_grid("2x2"),
    )
    parser.add_argument("--tile-buffer-m", type=float, default=30.0)
    parser.add_argument("--max-cache-mb", type=float, default=200.0)
    parser.add_argument("--gzip-cache", dest="gzip_cache", action="store_true", default=True)
    parser.add_argument("--no-gzip-cache", dest="gzip_cache", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def compute_bounds(features: Iterable[Dict[str, object]]) -> Optional[Tuple[float, float, float, float]]:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception:  # pragma: no cover - defensive
            continue
        if geom.is_empty:
            continue
        bounds = geom.bounds
        minx = min(minx, bounds[0])
        miny = min(miny, bounds[1])
        maxx = max(maxx, bounds[2])
        maxy = max(maxy, bounds[3])
    if minx is math.inf:
        return None
    return (minx, miny, maxx, maxy)


def summarize_ids(features: Iterable[Dict[str, object]]) -> Tuple[int, List[str]]:
    ids = [_stable_feature_id(feature) for feature in features]
    return len(ids), ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    bbox_tuple = tuple(args.bbox)
    bbox_str = ",".join(f"{value:.6f}" for value in bbox_tuple)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fetching reference payload for %s", bbox_str)
    base_provider = OvertureBuildingsProvider(
        base_url=args.overture_endpoint,
        theme=args.overture_theme,
        search_radius_m=0.0,
        cache_quantization_m=0.0,
        match_distance_m=0.0,
        sleep_between_requests=0.0,
        limit=args.limit,
        include_fields=args.include_fields,
        category_fields=args.category_fields,
        name_fields=args.name_fields,
        timeout=0.0,
        cache_dir=cache_dir,
        cache_only=args.cache_only,
        tile_config=(1, 1),
        tile_buffer_m=args.tile_buffer_m,
        max_cache_mb=args.max_cache_mb,
        gzip_cache=args.gzip_cache,
    )
    big_payload = base_provider._call_overture_cli_paged(bbox_str)
    big_features_raw = big_payload.get("features") if isinstance(big_payload, dict) else []
    if not isinstance(big_features_raw, list):
        big_features_raw = []
    big_features = _deduplicate_features(big_features_raw)

    LOGGER.info("Fetching tiled payload for %s", bbox_str)
    tile_provider = OvertureBuildingsProvider(
        base_url=args.overture_endpoint,
        theme=args.overture_theme,
        search_radius_m=0.0,
        cache_quantization_m=0.0,
        match_distance_m=0.0,
        sleep_between_requests=0.0,
        limit=args.limit,
        include_fields=args.include_fields,
        category_fields=args.category_fields,
        name_fields=args.name_fields,
        timeout=0.0,
        cache_dir=cache_dir,
        cache_only=args.cache_only,
        tile_config=args.tile,
        tile_buffer_m=args.tile_buffer_m,
        max_cache_mb=args.max_cache_mb,
        gzip_cache=args.gzip_cache,
    )
    tile_payload = tile_provider._call_overture_cli_tiled(bbox_tuple)
    tile_features_raw = tile_payload.get("features") if isinstance(tile_payload, dict) else []
    if not isinstance(tile_features_raw, list):
        tile_features_raw = []
    tile_features = _deduplicate_features(tile_features_raw)

    big_count_raw = len(big_features_raw)
    big_count = len(big_features)
    tile_count_raw = len(tile_features_raw)
    tile_count = len(tile_features)

    big_bounds = compute_bounds(big_features)
    tile_bounds = compute_bounds(tile_features)

    _, big_ids = summarize_ids(big_features)
    _, tile_ids = summarize_ids(tile_features)
    big_id_set = set(big_ids)
    tile_id_set = set(tile_ids)
    missing_ids = sorted(big_id_set - tile_id_set)
    extra_ids = sorted(tile_id_set - big_id_set)

    comparison = {
        "bbox": bbox_tuple,
        "tile_grid": args.tile,
        "tile_buffer_m": args.tile_buffer_m,
        "big": {
            "raw_features": big_count_raw,
            "deduplicated": big_count,
            "bounds": big_bounds,
        },
        "tiled": {
            "raw_features": tile_count_raw,
            "deduplicated": tile_count,
            "bounds": tile_bounds,
            "duplicates_removed": tile_count_raw - tile_count,
        },
        "missing_ids": missing_ids[:20],
        "extra_ids": extra_ids[:20],
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
    }

    print(json.dumps(comparison, indent=2))
    return 0 if not missing_ids and not extra_ids else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
