#!/usr/bin/env python3
"""Generate semantic dataset enriched with OSM + Overture data only."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.osm import (  # noqa: E402
    CLASS_TO_ID,
    DEFAULT_OVERPASS_FALLBACKS,
    ROAD_WIDTHS,
    build_building_features,
    classify_semantic,
    compute_bbox,
    download_osm,
    element_to_geometry,
    get_requests,
    latlon_to_local_projector,
    local_to_latlon_projector,
    rasterize_semantics,
)
from tools.osm import generate_semantic_dataset as osm_generate  # noqa: E402

from tools.multisource.generate_semantic_dataset_enriched import (  # noqa: E402
    OvertureBuildingsProvider,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PROVIDER_RADIUS = 80.0
DEFAULT_PROVIDER_CACHE_QUANTIZATION = 25.0
DEFAULT_MATCH_DISTANCE = 45.0
DEFAULT_REQUEST_SLEEP = 0.2

DEFAULT_OVERTURE_INCLUDE_FIELDS = [
    "names",
    "categories",
    "addresses",
    "building",
    "building.levels",
    "height",
]


@dataclass
class Stage1Result:
    primary: str
    secondary: str
    tertiary: str
    confidence: float
    conflict: bool


def parse_feature_filter(raw: str) -> Sequence[str]:
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def parse_tile_grid(raw: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", raw)
    if not match:
        raise argparse.ArgumentTypeError("--tile must be of the form NxM, e.g. 2x2 or 3x3")
    nx, ny = int(match.group(1)), int(match.group(2))
    if nx <= 0 or ny <= 0:
        raise argparse.ArgumentTypeError("--tile dimensions must be positive integers")
    return nx, ny


def _source_keys(source_tags: Iterable[str]) -> Tuple[set, Dict[str, List[str]]]:
    keys = set()
    values: Dict[str, List[str]] = {}
    for tag in source_tags or []:
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        keys.add(key)
        values.setdefault(key, []).append(value)
    return keys, values


def stage1_from_osm(properties: Dict[str, Any]) -> Stage1Result:
    primary = properties.get("primary_label") or ""
    secondary = properties.get("secondary_label") or ""
    tertiary = properties.get("tertiary_label") or ""
    confidence = 0.0
    if primary:
        confidence += 0.5
    if secondary:
        confidence += 0.2
    if tertiary:
        confidence += 0.1
    if properties.get("name"):
        confidence += 0.1
    if properties.get("building_levels") or properties.get("height"):
        confidence += 0.1
    confidence = min(confidence, 1.0)

    keys, values = _source_keys(properties.get("source_tags", []))
    conflict = False
    if len(keys) > 1:
        conflict = True
    else:
        for key, val_list in values.items():
            if len(set(val_list)) > 1:
                conflict = True
                break
    return Stage1Result(primary=primary, secondary=secondary, tertiary=tertiary, confidence=confidence, conflict=conflict)


def combine_labels(osm_props: Dict[str, Any], provider_types: Sequence[str]) -> Tuple[str, str, str, str, str]:
    osm_primary = osm_props.get("primary_label") or ""
    osm_secondary = osm_props.get("secondary_label") or ""
    osm_tertiary = osm_props.get("tertiary_label") or ""

    provider_clean = [ptype.replace("_", " ") for ptype in provider_types if ptype]
    primary = osm_primary or (provider_clean[0] if provider_clean else "")
    secondary = osm_secondary
    if provider_clean:
        main = provider_clean[0]
        if secondary and main not in secondary:
            secondary = f"{secondary}|{main}"
        elif not secondary:
            secondary = main
    tertiary = osm_tertiary
    additional = provider_clean[1] if len(provider_clean) > 1 else ""
    if additional:
        if tertiary and additional not in tertiary:
            tertiary = f"{tertiary}|{additional}"
        elif not tertiary:
            tertiary = additional

    category_path = "|".join(filter(None, [primary, secondary, tertiary]))
    provenance = "osm" if osm_primary else "provider"
    if osm_primary and provider_clean:
        provenance = "osm+provider"
    return primary, secondary, tertiary, category_path, provenance


def load_osm_features(
    lat: float,
    lon: float,
    radius_m: float,
    overpass_url: str,
    fallback_overpass: Sequence[str],
    resolution: float,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bbox = compute_bbox(lat, lon, radius_m)
    projector = latlon_to_local_projector(lat, lon)
    inverse_projector = local_to_latlon_projector(lat, lon)

    requests_module = get_requests()
    osm_generate.DEFAULT_OVERPASS_FALLBACKS = list(fallback_overpass)

    LOGGER.info("Downloading OSM data from %s", overpass_url)
    elements = download_osm(bbox, overpass_url, requests_module)
    LOGGER.info("Received %d elements", len(elements))

    class_geometries: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {c: [] for c in CLASS_TO_ID}
    buildings: List[dict] = []

    for element in elements:
        geom = element_to_geometry(element)
        if geom is None:
            continue
        tags = element.get("tags", {})
        projected = projector(geom)
        if projected.is_empty:
            continue
        semantic_class = classify_semantic(tags)
        if semantic_class in {"road", "traffic_road"}:
            width = ROAD_WIDTHS["traffic" if semantic_class == "traffic_road" else "default"]
            class_geometries[semantic_class].append((projected, {"width_m": width}))
        elif semantic_class and semantic_class in class_geometries:
            class_geometries[semantic_class].append((projected, {}))
        if "building" in tags:
            buildings.append(element)

    raster_output = output_dir / "semantic_map.npy"
    rasterize_semantics(class_geometries, resolution, radius_m, raster_output)

    features = build_building_features(buildings, projector, inverse_projector, radius_m)
    for feature, element in zip(features, buildings):
        props = feature.setdefault("properties", {})
        props["osm_id"] = element.get("id")
        props["osm_type"] = element.get("type", "way")
        props["osm_tags_raw"] = element.get("tags", {})

    metadata = {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "resolution_m": resolution,
        "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
        "semantic_map_path": str(raster_output),
    }
    return features, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enriched dataset using only OSM + Overture data.")
    parser.add_argument("lat", type=float, help="Center latitude")
    parser.add_argument("lon", type=float, help="Center longitude")
    parser.add_argument("radius", type=float, help="Half side length of square area in meters")
    parser.add_argument("--resolution", type=float, default=1.0, help="Raster resolution in meters per pixel")
    parser.add_argument("--output", type=Path, default=Path("./semantic_dataset_osm_overture_full"), help="Output directory")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api/interpreter", help="Primary Overpass endpoint")
    parser.add_argument(
        "--fallback-overpass",
        nargs="*",
        default=DEFAULT_OVERPASS_FALLBACKS,
        help="Fallback Overpass endpoints",
    )
    parser.add_argument("--feature-filter", default="", help="Optional comma-separated list of OSM source tags to keep")
    parser.add_argument("--min-area-m2", type=float, default=0.0, help="Minimum footprint area to keep (default keeps all)")
    parser.add_argument("--provider-radius", type=float, default=DEFAULT_PROVIDER_RADIUS)
    parser.add_argument("--provider-cache-quantization", type=float, default=DEFAULT_PROVIDER_CACHE_QUANTIZATION)
    parser.add_argument("--match-distance", type=float, default=DEFAULT_MATCH_DISTANCE)
    parser.add_argument("--request-sleep", type=float, default=DEFAULT_REQUEST_SLEEP)
    parser.add_argument("--overture-endpoint", default="https://api.overturemaps.org/places/v1/places")
    parser.add_argument("--overture-theme", default="buildings")
    parser.add_argument("--overture-limit", type=int, default=50)
    parser.add_argument(
        "--overture-include-fields",
        nargs="*",
        default=DEFAULT_OVERTURE_INCLUDE_FIELDS,
    )
    parser.add_argument(
        "--overture-category-fields",
        nargs="*",
        default=["categories", "category", "function", "building.use", "building.function"],
    )
    parser.add_argument(
        "--overture-name-fields",
        nargs="*",
        default=["name", "names.primary", "display.name", "displayName.text"],
    )
    parser.add_argument(
        "--overture-timeout-s",
        "--overture-timeout",
        dest="overture_timeout_s",
        type=int,
        default=None,
        help="Timeout in seconds for overture CLI calls (overrides OVERTURE_TIMEOUT_S env).",
    )
    parser.add_argument(
        "--tile",
        type=parse_tile_grid,
        default=parse_tile_grid("2x2"),
        help="Tile grid for large bbox fetching (e.g. 2x2, 3x3).",
    )
    parser.add_argument(
        "--tile-buffer-m",
        type=float,
        default=30.0,
        help="Buffer in metres added around each tile during Overture fetches (default: 30m)",
    )
    parser.add_argument(
        "--max-cache-mb",
        type=float,
        default=200.0,
        help="Maximum accepted cache size before forcing a refresh (default: 200 MB)",
    )
    parser.add_argument(
        "--gzip-cache",
        dest="gzip_cache",
        action="store_true",
        default=True,
        help="Store cached Overture payloads as .json.gz (default: enabled)",
    )
    parser.add_argument(
        "--no-gzip-cache",
        dest="gzip_cache",
        action="store_false",
        help="Disable gzip compression for cached Overture payloads",
    )
    parser.add_argument(
        "--overture-cache-dir",
        type=Path,
        default=Path("data/overture_cache"),
        help="Directory used to cache downloaded Overture payloads",
    )
    parser.add_argument(
        "--overture-cache-only",
        action="store_true",
        help="Skip Overture network calls and rely solely on cached responses",
    )
    parser.add_argument(
        "--overture-prefetch-radius",
        type=float,
        default=None,
        help="Radius in meters for shared Overture downloads (defaults to dataset radius)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.providers = "osm,overture"
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    args.output.mkdir(parents=True, exist_ok=True)

    features, metadata = load_osm_features(
        args.lat,
        args.lon,
        args.radius,
        args.overpass_url,
        args.fallback_overpass,
        args.resolution,
        args.output,
    )

    allowed_filter = set(parse_feature_filter(args.feature_filter))

    overture_provider: Optional[OvertureBuildingsProvider] = None
    overture_matches = 0
    overture_errors = 0
    skipped_filter = 0

    try:
        overture_cache_dir = args.overture_cache_dir.expanduser().resolve()
        overture_cache_dir.mkdir(parents=True, exist_ok=True)
        overture_provider = OvertureBuildingsProvider(
            base_url=args.overture_endpoint,
            theme=args.overture_theme,
            search_radius_m=args.provider_radius,
            cache_quantization_m=args.provider_cache_quantization,
            match_distance_m=args.match_distance,
            sleep_between_requests=args.request_sleep,
            limit=args.overture_limit,
            include_fields=args.overture_include_fields,
            category_fields=args.overture_category_fields,
            name_fields=args.overture_name_fields,
            timeout=float(args.overture_timeout_s or 0),
            cache_dir=overture_cache_dir,
            cache_only=args.overture_cache_only,
            tile_config=args.tile,
            tile_buffer_m=args.tile_buffer_m,
            max_cache_mb=args.max_cache_mb,
            gzip_cache=args.gzip_cache,
            cli_timeout_s=args.overture_timeout_s,
        )
    except RuntimeError as exc:
        LOGGER.error("Unable to initialise Overture provider: %s", exc)
        overture_provider = None
    else:
        prefetch_radius = args.overture_prefetch_radius if args.overture_prefetch_radius is not None else args.radius
        overture_provider.configure_prefetch_region(args.lat, args.lon, prefetch_radius)

    filtered_features: List[Dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        area = props.get("area_m2", 0.0)
        if area and area < args.min_area_m2:
            skipped_filter += 1
            continue
        tag_keys, _ = _source_keys(props.get("source_tags", []))
        if allowed_filter and not (tag_keys & allowed_filter):
            skipped_filter += 1
            continue
        filtered_features.append(feature)

    LOGGER.info("Processing %d filtered features (skipped %d by filter)", len(filtered_features), skipped_filter)

    enriched_features: List[Dict[str, Any]] = []

    for feature in filtered_features:
        feature = dict(feature)
        props = dict(feature.get("properties", {}))
        stage1 = stage1_from_osm(props)
        props["stage1_confidence"] = stage1.confidence
        props["stage1_conflict"] = stage1.conflict

        overture_data = None
        centroid_lat = props.get("centroid_lat")
        centroid_lon = props.get("centroid_lon")

        if overture_provider and centroid_lat is not None and centroid_lon is not None:
            try:
                overture_data = overture_provider.lookup(centroid_lat, centroid_lon, feature)
            except Exception as exc:
                LOGGER.warning(
                    "Overture lookup failed near %.6f, %.6f: %s",
                    centroid_lat,
                    centroid_lon,
                    exc,
                )
                overture_errors += 1
                overture_data = None
            else:
                if overture_data:
                    overture_matches += 1

        provider_types: List[str] = []
        provider_sources: List[str] = []
        provider_confidence = stage1.confidence
        provider_place_id = props.get("provider_place_id")

        if overture_data:
            provider_sources.append("overture_buildings")
            overture_categories = [str(category) for category in overture_data.categories if category]
            for category in overture_categories:
                if category not in provider_types:
                    provider_types.append(category)
            provider_confidence = max(provider_confidence, overture_data.confidence or 0.0)
            if overture_data.place_id and not provider_place_id:
                provider_place_id = overture_data.place_id
            props.update(
                {
                    "overture_place_id": overture_data.place_id,
                    "overture_categories": overture_categories,
                    "overture_distance_m": overture_data.distance_m,
                    "overture_confidence": overture_data.confidence,
                    "overture_raw": overture_data.raw,
                }
            )
        else:
            props.setdefault("overture_categories", [])

        (
            enriched_primary,
            enriched_secondary,
            enriched_tertiary,
            category_path,
            category_provenance,
        ) = combine_labels(props, provider_types)

        enriched_name = props.get("name") or ""
        enriched_name_provenance = "osm" if enriched_name else ""
        if overture_data and overture_data.name:
            enriched_name = overture_data.name
            enriched_name_provenance = "overture_buildings"

        props.update(
            {
                "enriched_primary_label": enriched_primary,
                "enriched_secondary_label": enriched_secondary,
                "enriched_tertiary_label": enriched_tertiary,
                "enriched_category_path": category_path,
                "enriched_category_provenance": category_provenance,
                "enriched_name": enriched_name,
                "enriched_name_provenance": enriched_name_provenance,
                "provider_place_id": provider_place_id,
                "provider_types": provider_types,
                "provider_sources": provider_sources,
                "provider_confidence": provider_confidence,
            }
        )

        feature["properties"] = props
        enriched_features.append(feature)

    collection = {"type": "FeatureCollection", "features": enriched_features}
    output_geojson = args.output / "buildings_enriched_osm_overture.geojson"
    with output_geojson.open("w", encoding="utf-8") as handle:
        json.dump(collection, handle, ensure_ascii=False, indent=2)

    metadata["buildings_geojson"] = str(output_geojson)
    metadata["provider"] = {
        "overture_enabled": overture_provider is not None,
        "overture_matches": overture_matches,
        "overture_errors": overture_errors,
    }

    metadata_path = args.output / "metadata_osm_overture.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if overture_provider:
        overture_provider.close()

    LOGGER.info(
        "Finished. overture_matches=%d overture_errors=%d skipped_filter=%d",
        overture_matches,
        overture_errors,
        skipped_filter,
    )


if __name__ == "__main__":
    main()

