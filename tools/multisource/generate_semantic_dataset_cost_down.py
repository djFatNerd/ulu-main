#!/usr/bin/env python3
"""Generate semantic dataset with cost-optimized Google enrichment."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tools.multisource.providers.google_provider import GooglePlacesDetailsClient
from tools.osm import (
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
from tools.osm import generate_semantic_dataset as osm_generate

from tools.multisource.generate_semantic_dataset_enriched import (
    OvertureBuildingsProvider,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_PROVIDER_FIELDS = {"place_id", "name", "types", "vicinity"}

DEFAULT_PROVIDER_RADIUS = 50.0
DEFAULT_PROVIDER_CACHE_QUANTIZATION = 25.0
DEFAULT_MATCH_DISTANCE = 35.0
DEFAULT_REQUEST_SLEEP = 0.2


@dataclass
class Stage1Result:
    primary: str
    secondary: str
    tertiary: str
    confidence: float
    conflict: bool


def _meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_factor = 111_320.0
    lon_factor = 111_320.0 * math.cos(math.radians(lat))
    return lat_factor, max(lon_factor, 1e-6)


def _to_local(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    lat_scale, lon_scale = _meters_per_degree(origin_lat)
    x = (lon - origin_lon) * lon_scale
    y = (lat - origin_lat) * lat_scale
    return x, y


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def parse_feature_filter(raw: str) -> Sequence[str]:
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


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


def should_use_google(stage1: Stage1Result, threshold: float) -> bool:
    return stage1.confidence < threshold or stage1.conflict


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
    metadata = {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "resolution_m": resolution,
        "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
        "semantic_map_path": str(raster_output),
    }
    return features, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enriched dataset with minimal Google usage.")
    parser.add_argument("lat", type=float, help="Center latitude")
    parser.add_argument("lon", type=float, help="Center longitude")
    parser.add_argument("radius", type=float, help="Half side length of square area in meters")
    parser.add_argument("--resolution", type=float, default=1.0, help="Raster resolution in meters per pixel")
    parser.add_argument("--output", type=Path, default=Path("./semantic_dataset_cost_down"), help="Output directory")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api/interpreter", help="Primary Overpass endpoint")
    parser.add_argument(
        "--fallback-overpass",
        nargs="*",
        default=DEFAULT_OVERPASS_FALLBACKS,
        help="Fallback Overpass endpoints",
    )
    parser.add_argument("--providers", default="osm,overture", help="Comma-separated provider list")
    parser.add_argument("--google-api-key", default=None, help="Google Maps API key")
    parser.add_argument("--google-mode", default="essentials")
    parser.add_argument("--google-fields", default="place_id,name,types,vicinity")
    parser.add_argument("--google-disable-photos", type=lambda v: str(v).lower() not in {"false", "0", "no"}, default=True)
    parser.add_argument("--feature-filter", default="building,amenity")
    parser.add_argument("--min-area-m2", type=float, default=50.0)
    parser.add_argument("--grid-size-m", type=float, default=50.0)
    parser.add_argument("--propagate-radius-m", type=float, default=25.0)
    parser.add_argument("--google-budget-requests", type=int, default=500)
    parser.add_argument("--google-qps-max", type=float, default=0.9)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.6)
    parser.add_argument("--max-distance-m", type=float, default=25.0)
    parser.add_argument("--unit-price-details-min", type=float, default=20.0)
    parser.add_argument("--google-cache-dir", type=Path, default=Path("~/.cache/ulu/google").expanduser())
    parser.add_argument("--place-id-property", default="provider_place_id", help="Feature property to use for Google place IDs")
    parser.add_argument("--provider-radius", type=float, default=DEFAULT_PROVIDER_RADIUS)
    parser.add_argument("--provider-cache-quantization", type=float, default=DEFAULT_PROVIDER_CACHE_QUANTIZATION)
    parser.add_argument("--match-distance", type=float, default=DEFAULT_MATCH_DISTANCE)
    parser.add_argument("--request-sleep", type=float, default=DEFAULT_REQUEST_SLEEP)
    parser.add_argument("--overture-endpoint", default="https://api.overturemaps.org/places/v1/places")
    parser.add_argument("--overture-theme", default="buildings")
    parser.add_argument("--overture-limit", type=int, default=25)
    parser.add_argument(
        "--overture-include-fields",
        nargs="*",
        default=["names", "categories", "addresses"],
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
    parser.add_argument("--overture-timeout", type=float, default=20.0)
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
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    providers = [token.strip().lower() for token in args.providers.split(",") if token.strip()]
    enable_google = "google" in providers and bool(args.google_api_key)

    if enable_google and args.google_mode.lower() != "essentials":
        LOGGER.warning("Overriding google_mode to essentials; only minimal fields are permitted.")
    google_fields = tuple(field.strip() for field in args.google_fields.split(",") if field.strip())
    if set(google_fields) - ALLOWED_PROVIDER_FIELDS:
        raise ValueError("google_fields may only include place_id,name,types,vicinity")

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

    google_client: Optional[GooglePlacesDetailsClient] = None
    if enable_google:
        args.google_cache_dir = args.google_cache_dir.expanduser()
        google_client = GooglePlacesDetailsClient(
            api_key=args.google_api_key,
            cache_dir=args.google_cache_dir,
            fields=google_fields,
            qps_max=args.google_qps_max,
            budget_requests=args.google_budget_requests,
            disable_photos=args.google_disable_photos,
            unit_price_usd_per_k=args.unit_price_details_min,
        )
    elif "google" in providers and not args.google_api_key:
        LOGGER.warning("Google provider requested but no API key provided; skipping Google enrichment.")

    overture_provider: Optional[OvertureBuildingsProvider] = None
    overture_enabled = "overture" in providers
    overture_matches = 0
    overture_errors = 0

    if overture_enabled:
        overture_cache_dir = args.overture_cache_dir.expanduser().resolve()
        overture_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
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
                timeout=args.overture_timeout,
                cache_dir=overture_cache_dir,
                cache_only=args.overture_cache_only,
            )
        except RuntimeError as exc:
            LOGGER.error("Unable to initialise Overture provider: %s", exc)
            overture_provider = None
            overture_enabled = False
        else:
            prefetch_radius = args.overture_prefetch_radius if args.overture_prefetch_radius is not None else args.radius
            overture_provider.configure_prefetch_region(args.lat, args.lon, prefetch_radius)

    filtered_features: List[Dict[str, Any]] = []
    skipped_filter = 0
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

    grid_results: Dict[Tuple[int, int], Optional[Tuple[Any, float, float]]] = {}
    grid_calls: Set[Tuple[int, int]] = set()
    requested_place_ids: set = set()

    enriched_features: List[Dict[str, Any]] = []
    matched = 0
    skipped_confidence = 0
    skipped_no_place_id = 0

    start_time = time.time()
    log_interval = 200

    for index, feature in enumerate(filtered_features, start=1):
        feature = dict(feature)
        props = dict(feature.get("properties", {}))
        stage1 = stage1_from_osm(props)
        props["stage1_confidence"] = stage1.confidence
        props["stage1_conflict"] = stage1.conflict
        provider_data = None

        overture_data = None

        centroid_lat = props.get("centroid_lat")
        centroid_lon = props.get("centroid_lon")
        grid_key = None
        if centroid_lat is not None and centroid_lon is not None and args.grid_size_m > 0:
            local_x, local_y = _to_local(centroid_lat, centroid_lon, args.lat, args.lon)
            grid_key = (int(local_x // args.grid_size_m), int(local_y // args.grid_size_m))
            cached_cell = grid_results.get(grid_key)
            if cached_cell:
                cached_result, ref_lat, ref_lon = cached_cell
                if cached_result is not None:
                    distance = haversine_distance_m(ref_lat, ref_lon, centroid_lat, centroid_lon)
                    if distance <= args.propagate_radius_m:
                        provider_data = cached_result
                else:
                    provider_data = None

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

        if provider_data is None and google_client and should_use_google(stage1, args.uncertainty_threshold):
            place_id = props.get(args.place_id_property) or props.get("place_id")
            if not place_id:
                skipped_no_place_id += 1
            else:
                if grid_key is not None and grid_key in grid_calls:
                    provider_data = None
                elif place_id in requested_place_ids:
                    provider_data = google_client.get_details(place_id, lat=centroid_lat, lon=centroid_lon)
                    if grid_key is not None and provider_data is not None:
                        grid_calls.add(grid_key)
                else:
                    requested_place_ids.add(place_id)
                    result = google_client.get_details(place_id, lat=centroid_lat, lon=centroid_lon)
                    if grid_key is not None:
                        grid_calls.add(grid_key)
                    if result is None:
                        provider_data = None
                    else:
                        provider_data = result
                        matched += 1
                        if grid_key is not None:
                            grid_results[grid_key] = (provider_data, centroid_lat, centroid_lon)
        elif provider_data is None and not should_use_google(stage1, args.uncertainty_threshold):
            skipped_confidence += 1

        provider_types: List[str] = []
        provider_sources: List[str] = []
        provider_confidence = stage1.confidence
        provider_place_id = props.get(args.place_id_property)
        provider_vicinity: Optional[str] = None

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

        if provider_data:
            for provider_type in provider_data.types:
                if provider_type and provider_type not in provider_types:
                    provider_types.append(provider_type)
            if "google_places" not in provider_sources:
                provider_sources.append("google_places")
            provider_confidence = max(provider_confidence, 0.8)
            provider_place_id = provider_data.place_id or provider_place_id
            provider_vicinity = provider_data.vicinity

        (enriched_primary, enriched_secondary, enriched_tertiary, category_path, category_provenance) = combine_labels(
            props, provider_types
        )

        enriched_name = props.get("name") or ""
        enriched_name_provenance = "osm" if enriched_name else ""
        if overture_data and overture_data.name:
            enriched_name = overture_data.name
            enriched_name_provenance = "overture_buildings"
        if provider_data and provider_data.name:
            enriched_name = provider_data.name
            enriched_name_provenance = "google_places"

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
                "provider_vicinity": provider_vicinity,
                "provider_types": provider_types,
                "provider_sources": provider_sources,
                "provider_confidence": provider_confidence,
            }
        )

        feature["properties"] = props
        enriched_features.append(feature)

        if index % log_interval == 0 or index == len(filtered_features):
            elapsed = max(time.time() - start_time, 1e-6)
            qps = (google_client.request_count / elapsed) if google_client else 0.0
            requests_made = google_client.request_count if google_client else 0
            LOGGER.info(
                "features_processed=%d/%d matches=%d provider_requests=%d qps=%.2f",
                index,
                len(filtered_features),
                matched,
                requests_made,
                qps,
            )
            if google_client:
                LOGGER.info(
                    "google_calls_details_min=%d est_cost_usd=%.2f",
                    google_client.request_count,
                    google_client.estimate_cost(),
                )

    collection = {"type": "FeatureCollection", "features": enriched_features}
    output_geojson = args.output / "buildings_enriched.geojson"
    with output_geojson.open("w", encoding="utf-8") as handle:
        json.dump(collection, handle, ensure_ascii=False)

    metadata["buildings_geojson"] = str(output_geojson)
    if google_client:
        total = max(google_client.cache_hits + google_client.cache_misses, 1)
        cache_hit_rate = google_client.cache_hits / total
    else:
        cache_hit_rate = 1.0
    metadata.update(
        {
            "provider": {
                "google_calls_details_min": google_client.request_count if google_client else 0,
                "google_cache_hits": google_client.cache_hits if google_client else 0,
                "google_cache_misses": google_client.cache_misses if google_client else 0,
                "cache_hit_rate": cache_hit_rate,
                "skipped_budget": google_client.skipped_budget if google_client else 0,
                "skipped_filter": skipped_filter,
                "skipped_confidence": skipped_confidence,
                "skipped_place_id": skipped_no_place_id,
                "provider_requests": google_client.request_count if google_client else 0,
                "provider_matches": matched,
                "overture_enabled": overture_enabled,
                "overture_matches": overture_matches,
                "overture_errors": overture_errors,
            }
        }
    )

    with (args.output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if google_client:
        google_client.close()
    if overture_provider:
        overture_provider.close()

    LOGGER.info(
        "Finished. google_calls_details_min=%d cache_hit_rate=%.2f skipped_budget=%d skipped_filter=%d skipped_confidence=%d",
        google_client.request_count if google_client else 0,
        cache_hit_rate,
        google_client.skipped_budget if google_client else 0,
        skipped_filter,
        skipped_confidence,
    )


if __name__ == "__main__":
    main()
