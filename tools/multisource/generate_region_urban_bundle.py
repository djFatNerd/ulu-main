#!/usr/bin/env python3
"""Generate a fully enriched regional bundle using OSM, Overture, and Google data."""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.multisource.generate_semantic_dataset_enriched import (
    DEFAULT_OVERPASS_FALLBACKS,
    CompositeProvider,
    GooglePlacesProvider,
    NullProvider,
    OvertureBuildingsProvider,
    ProviderBase,
    run_generation,
)

LOGGER = logging.getLogger(__name__)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _summarize(values: Iterable[float]) -> Optional[Dict[str, float]]:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return None
    return {
        "min": min(numeric),
        "max": max(numeric),
        "mean": mean(numeric),
        "median": median(numeric),
    }


def _build_building_catalog(features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for feature in features:
        properties = dict(feature.get("properties", {}))
        record = {
            "id": feature.get("id"),
            "centroid": {
                "lat": properties.get("centroid_lat"),
                "lon": properties.get("centroid_lon"),
            },
            "area_m2": properties.get("area_m2"),
            "osm_labels": {
                "primary": properties.get("primary_label"),
                "secondary": properties.get("secondary_label"),
                "tertiary": properties.get("tertiary_label"),
            },
            "enriched_labels": {
                "primary": properties.get("enriched_primary_label"),
                "secondary": properties.get("enriched_secondary_label"),
                "tertiary": properties.get("enriched_tertiary_label"),
                "category_path": properties.get("enriched_category_path"),
                "provenance": properties.get("enriched_category_provenance"),
            },
            "name": {
                "value": properties.get("enriched_name") or properties.get("name"),
                "provenance": properties.get("enriched_name_provenance"),
            },
            "ratings": {
                "value": properties.get("enriched_rating"),
                "count": properties.get("enriched_rating_count"),
                "provenance": properties.get("enriched_rating_provenance"),
            },
            "opening_hours": {
                "value": properties.get("enriched_opening_hours"),
                "provenance": properties.get("enriched_opening_hours_provenance"),
            },
            "provider": {
                "place_id": properties.get("provider_place_id"),
                "distance_m": properties.get("provider_distance_m"),
                "confidence": properties.get("provider_confidence"),
                "categories": properties.get("provider_categories"),
                "categories_provenance": properties.get("provider_categories_provenance"),
                "place_id_provenance": properties.get("provider_place_id_provenance"),
                "sources": properties.get("provider_sources"),
                "raw": properties.get("provider_raw"),
            },
            "geometry": feature.get("geometry"),
            "attributes": {k: v for k, v in properties.items() if k not in {
                "centroid_lat",
                "centroid_lon",
                "area_m2",
                "primary_label",
                "secondary_label",
                "tertiary_label",
                "enriched_primary_label",
                "enriched_secondary_label",
                "enriched_tertiary_label",
                "enriched_category_path",
                "enriched_category_provenance",
                "enriched_name",
                "enriched_name_provenance",
                "enriched_rating",
                "enriched_rating_provenance",
                "enriched_rating_count",
                "enriched_opening_hours",
                "enriched_opening_hours_provenance",
                "provider_place_id",
                "provider_distance_m",
                "provider_confidence",
                "provider_categories",
                "provider_categories_provenance",
                "provider_place_id_provenance",
                "provider_sources",
                "provider_raw",
            }},
        }
        catalog.append(record)
    return catalog


def _compute_region_summary(features: Sequence[Dict[str, Any]], radius_m: float) -> Dict[str, Any]:
    area_values: List[float] = []
    level_values: List[float] = []
    height_values: List[float] = []
    rating_values: List[float] = []
    categories = Counter()

    for feature in features:
        props = feature.get("properties", {})
        area = _as_float(props.get("area_m2"))
        if area is not None:
            area_values.append(area)
        levels = props.get("building_levels") or props.get("levels")
        level_value = _as_float(levels)
        if level_value is not None:
            level_values.append(level_value)
        height = props.get("height")
        height_value = _as_float(height)
        if height_value is not None:
            height_values.append(height_value)
        rating = props.get("enriched_rating")
        rating_value = _as_float(rating)
        if rating_value is not None:
            rating_values.append(rating_value)
        primary_label = props.get("enriched_primary_label") or props.get("primary_label")
        if primary_label:
            categories[primary_label] += 1

    return {
        "region_area_m2": (2 * radius_m) ** 2,
        "building_area_stats": _summarize(area_values),
        "building_level_stats": _summarize(level_values),
        "building_height_stats": _summarize(height_values),
        "rating_stats": _summarize(rating_values),
        "dominant_categories": [
            {"label": label, "count": count}
            for label, count in categories.most_common()
        ],
    }


def _build_provider(
    lat: float,
    lon: float,
    radius: float,
    google_api_key: Optional[str],
    provider_radius: float,
    match_distance: float,
    request_sleep: float,
    overture_theme: str,
    overture_limit: int,
    overture_include_fields: Sequence[str],
    overture_category_fields: Sequence[str],
    overture_name_fields: Sequence[str],
    overture_timeout: float,
    overture_cache_dir: Path,
    overture_cache_only: bool,
    overture_prefetch_radius: Optional[float],
) -> ProviderBase:
    providers: List[ProviderBase] = []

    if google_api_key:
        providers.append(
            GooglePlacesProvider(
                api_key=google_api_key,
                search_radius_m=provider_radius,
                match_distance_m=match_distance,
                sleep_between_requests=request_sleep,
                cache_quantization_m=25.0,
            )
        )
    else:
        LOGGER.warning(
            "No Google API key supplied; proceeding without Google enrichment."
        )

    overture_provider = OvertureBuildingsProvider(
        base_url="https://api.overturemaps.org/",
        theme=overture_theme,
        search_radius_m=provider_radius,
        cache_quantization_m=25.0,
        match_distance_m=match_distance,
        sleep_between_requests=request_sleep,
        limit=overture_limit,
        include_fields=overture_include_fields,
        category_fields=overture_category_fields,
        name_fields=overture_name_fields,
        timeout=overture_timeout,
        cache_dir=overture_cache_dir,
        cache_only=overture_cache_only,
    )
    prefetch_radius = overture_prefetch_radius if overture_prefetch_radius is not None else radius
    overture_provider.configure_prefetch_region(lat, lon, prefetch_radius)
    providers.append(overture_provider)

    if not providers:
        return NullProvider()
    if len(providers) == 1:
        return providers[0]
    return CompositeProvider(providers, label="+".join([p.name for p in providers]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an enriched regional dataset with semantic map, building catalog, and metadata.",
    )
    parser.add_argument("lat", type=float, help="Region center latitude in decimal degrees")
    parser.add_argument("lon", type=float, help="Region center longitude in decimal degrees")
    parser.add_argument("radius", type=float, help="Half side length of the square region in meters")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Raster resolution for the semantic map in meters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/urban_region_bundle"),
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--google-api-key",
        default=None,
        help="Optional Google Maps Platform API key (otherwise reads GOOGLE_MAPS_API_KEY)",
    )
    parser.add_argument(
        "--provider-radius",
        type=float,
        default=75.0,
        help="Search radius in meters for enrichment lookups",
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=40.0,
        help="Maximum centroid distance in meters to accept a provider match",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.2,
        help="Seconds to wait between provider requests to respect rate limits",
    )
    parser.add_argument(
        "--overture-cache-dir",
        type=Path,
        default=Path("data/overture_cache"),
        help="Directory for caching Overture downloads",
    )
    parser.add_argument(
        "--overture-cache-only",
        action="store_true",
        help="Reuse cached Overture payloads without issuing network requests",
    )
    parser.add_argument(
        "--overture-prefetch-radius",
        type=float,
        default=None,
        help="Optional radius in meters for shared Overture downloads (defaults to dataset radius)",
    )
    parser.add_argument(
        "--overture-limit",
        type=int,
        default=25,
        help="Maximum number of features to retrieve per Overture lookup",
    )
    parser.add_argument(
        "--overture-theme",
        default="buildings",
        help="Overture dataset theme to query",
    )
    parser.add_argument(
        "--overture-include-fields",
        nargs="*",
        default=["names", "categories", "addresses"],
        help="Fields retained from Overture responses for cache keying",
    )
    parser.add_argument(
        "--overture-category-fields",
        nargs="*",
        default=["categories", "category", "function", "building.use", "building.function"],
        help="Property paths used to extract categories from Overture responses",
    )
    parser.add_argument(
        "--overture-name-fields",
        nargs="*",
        default=["name", "names.primary", "display.name", "displayName.text"],
        help="Property paths used to extract names from Overture responses",
    )
    parser.add_argument(
        "--overture-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for Overture CLI invocations",
    )
    parser.add_argument(
        "--overpass-url",
        default=os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter"),
        help="Overpass API endpoint used for OSM downloads",
    )
    parser.add_argument(
        "--fallback-overpass",
        nargs="*",
        default=None,
        help="Optional list of fallback Overpass endpoints",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    google_api_key = args.google_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    overture_cache_dir = args.overture_cache_dir.expanduser().resolve()
    overture_cache_dir.mkdir(parents=True, exist_ok=True)

    provider = _build_provider(
        lat=args.lat,
        lon=args.lon,
        radius=args.radius,
        google_api_key=google_api_key,
        provider_radius=args.provider_radius,
        match_distance=args.match_distance,
        request_sleep=args.request_sleep,
        overture_theme=args.overture_theme,
        overture_limit=args.overture_limit,
        overture_include_fields=args.overture_include_fields,
        overture_category_fields=args.overture_category_fields,
        overture_name_fields=args.overture_name_fields,
        overture_timeout=args.overture_timeout,
        overture_cache_dir=overture_cache_dir,
        overture_cache_only=args.overture_cache_only,
        overture_prefetch_radius=args.overture_prefetch_radius,
    )

    fallback_overpass: Sequence[str]
    if args.fallback_overpass:
        fallback_overpass = list(args.fallback_overpass)
    else:
        fallback_overpass = list(DEFAULT_OVERPASS_FALLBACKS)

    provider_mode = "osm+" + provider.name if not isinstance(provider, NullProvider) else "osm"

    try:
        metadata = run_generation(
            lat=args.lat,
            lon=args.lon,
            radius_m=args.radius,
            resolution=args.resolution,
            output_dir=output_dir,
            overpass_url=args.overpass_url,
            fallback_overpass=fallback_overpass,
            provider=provider,
            use_osm_labels=True,
            provider_mode=provider_mode,
        )
    finally:
        provider.close()

    buildings_geojson_path = Path(metadata["buildings_geojson"]).resolve()
    with buildings_geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    features = geojson.get("features", [])

    building_catalog = _build_building_catalog(features)
    buildings_json_path = output_dir / "buildings_enriched.json"
    with buildings_json_path.open("w", encoding="utf-8") as f:
        json.dump({"buildings": building_catalog}, f, ensure_ascii=False, indent=2)

    region_summary = _compute_region_summary(features, args.radius)

    metadata.update(
        {
            "buildings_json": str(buildings_json_path),
            "region_summary": region_summary,
            "data_sources": {
                "osm": True,
                "google_places": bool(google_api_key),
                "overture": True,
            },
        }
    )

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    LOGGER.info("Semantic map saved to %s", metadata.get("semantic_map_path"))
    LOGGER.info("Building catalog saved to %s", buildings_json_path)
    LOGGER.info("Metadata saved to %s", metadata_path)


if __name__ == "__main__":
    main()
