#!/usr/bin/env python3
"""Generate semantic datasets enriched with external provider metadata."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tools.osm.generate_semantic_dataset as osm_script
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
    rasterize_semantics,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    """Standardized representation of enrichment provider responses."""

    provider: str
    place_id: Optional[str]
    name: Optional[str]
    categories: Sequence[str]
    rating: Optional[float]
    rating_count: Optional[int]
    opening_hours_text: Optional[Sequence[str]]
    raw: Dict[str, Any]
    distance_m: Optional[float]
    confidence: float


class ProviderBase:
    """Base class for enrichment providers."""

    name = "base"

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - hook for future resources
        return None


class NullProvider(ProviderBase):
    """Provider that performs no enrichment."""

    name = "osm"

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        return None


class GooglePlacesProvider(ProviderBase):
    """Lookup enrichment metadata via the Google Places API."""

    name = "google_places"

    def __init__(
        self,
        api_key: str,
        search_radius_m: float,
        match_distance_m: float,
        sleep_between_requests: float,
    ) -> None:
        try:
            import googlemaps  # type: ignore
        except ImportError as exc:  # pragma: no cover - executed only when dependency missing
            raise RuntimeError(
                "googlemaps package is required for Google provider. Install it via requirements.txt."
            ) from exc

        self._client = googlemaps.Client(key=api_key)
        self._search_radius_m = search_radius_m
        self._match_distance_m = match_distance_m
        self._sleep_between_requests = sleep_between_requests

    def _throttle(self) -> None:
        if self._sleep_between_requests > 0:
            time.sleep(self._sleep_between_requests)

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        from googlemaps import exceptions as g_exceptions  # type: ignore

        params = {
            "location": (lat, lon),
            "radius": self._search_radius_m,
        }
        try:
            self._throttle()
            response = self._client.places_nearby(**params)
        except g_exceptions.ApiError as exc:
            LOGGER.warning("Google Places API error for %s: %s", feature.get("id"), exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unexpected Google Places error for %s: %s", feature.get("id"), exc)
            return None

        candidates = response.get("results", [])
        if not candidates:
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        best_distance: Optional[float] = None
        for candidate in candidates:
            geom = candidate.get("geometry", {})
            location = geom.get("location") or {}
            c_lat = location.get("lat")
            c_lon = location.get("lng")
            if c_lat is None or c_lon is None:
                continue
            distance = haversine_distance_m(lat, lon, c_lat, c_lon)
            if distance is None or distance > self._match_distance_m:
                continue
            if best_distance is None or distance < best_distance:
                best_candidate = candidate
                best_distance = distance

        if not best_candidate:
            return None

        details: Dict[str, Any] = {}
        place_id = best_candidate.get("place_id")
        if place_id:
            try:
                self._throttle()
                details_resp = self._client.place(
                    place_id,
                    fields=[
                        "name",
                        "types",
                        "rating",
                        "user_ratings_total",
                        "opening_hours",
                        "website",
                        "formatted_phone_number",
                        "business_status",
                    ],
                )
                details = details_resp.get("result", {}) if details_resp else {}
            except g_exceptions.ApiError as exc:
                LOGGER.warning("Google Place details error for %s: %s", place_id, exc)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Unexpected Google Place details error for %s: %s", place_id, exc)

        categories: Sequence[str] = (
            details.get("types")
            or best_candidate.get("types")
            or []
        )
        opening_hours: Optional[Sequence[str]] = None
        opening = details.get("opening_hours") or best_candidate.get("opening_hours")
        if isinstance(opening, dict):
            opening_hours = opening.get("weekday_text")
        elif isinstance(opening, list):
            opening_hours = opening

        rating = details.get("rating")
        if rating is not None:
            try:
                rating = float(rating)
            except (TypeError, ValueError):
                rating = None

        rating_count = details.get("user_ratings_total")
        if rating_count is not None:
            try:
                rating_count = int(rating_count)
            except (TypeError, ValueError):
                rating_count = None

        confidence = 0.0
        if best_distance is not None and self._match_distance_m > 0:
            confidence = max(0.0, 1.0 - (best_distance / self._match_distance_m))

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=details.get("name") or best_candidate.get("name"),
            categories=categories,
            rating=rating,
            rating_count=rating_count,
            opening_hours_text=opening_hours,
            raw={"candidate": best_candidate, "details": details},
            distance_m=best_distance,
            confidence=confidence,
        )


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """Compute approximate distance between two coordinates in meters."""

    for value in (lat1, lon1, lat2, lon2):
        if value is None:
            return None
    rad = math.radians
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371000.0 * c


def select_provider(args: argparse.Namespace) -> ProviderBase:
    if args.provider == "osm":
        return NullProvider()
    if args.provider in {"google", "osm_google"}:
        if not args.google_api_key:
            raise ValueError(
                "Google API key missing. Provide --google-api-key or set GOOGLE_MAPS_API_KEY."
            )
        return GooglePlacesProvider(
            api_key=args.google_api_key,
            search_radius_m=args.provider_radius,
            match_distance_m=args.match_distance,
            sleep_between_requests=args.request_sleep,
        )
    raise ValueError(f"Unknown provider {args.provider}")


def normalize_category(value: str) -> str:
    return value.replace("_", " ") if value else value


def combine_labels(
    osm_properties: Dict[str, Any],
    provider_categories: Sequence[str],
) -> Tuple[str, str, str, str, str]:
    osm_primary = osm_properties.get("primary_label") or ""
    osm_secondary = osm_properties.get("secondary_label") or ""
    osm_tertiary = osm_properties.get("tertiary_label") or ""

    provider_clean = [normalize_category(cat) for cat in provider_categories]

    primary = osm_primary or (provider_clean[0] if provider_clean else "")

    secondary = osm_secondary
    if provider_clean:
        provider_main = provider_clean[0]
        if secondary and provider_main and provider_main not in secondary:
            secondary = f"{secondary}|{provider_main}"
        elif not secondary:
            secondary = provider_main

    tertiary = osm_tertiary
    provider_secondary = provider_clean[1] if len(provider_clean) > 1 else ""
    if provider_secondary:
        if tertiary and provider_secondary not in tertiary:
            tertiary = f"{tertiary}|{provider_secondary}"
        elif not tertiary:
            tertiary = provider_secondary

    category_path = "-".join([label for label in [primary, secondary, tertiary] if label])
    if provider_clean:
        provenance = "osm+provider" if (osm_primary or osm_secondary or osm_tertiary) else "provider_only"
    elif primary or secondary or tertiary:
        provenance = "osm_only"
    else:
        provenance = "none"

    return primary, secondary, tertiary, category_path, provenance


def enrich_features(
    features: Iterable[Dict[str, Any]],
    provider: ProviderBase,
    use_osm_labels: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    matched = 0
    provider_requests = 0

    for feature in features:
        props = dict(feature.get("properties", {}))
        if use_osm_labels:
            osm_for_combination = props
        else:
            osm_for_combination = {
                **props,
                "primary_label": "",
                "secondary_label": "",
                "tertiary_label": "",
            }
        centroid_lat = props.get("centroid_lat")
        centroid_lon = props.get("centroid_lon")
        provider_result = None
        if centroid_lat is not None and centroid_lon is not None:
            provider_result = provider.lookup(centroid_lat, centroid_lon, feature)
            provider_requests += 1

        if provider_result:
            matched += 1
            provider_categories = list(provider_result.categories)
            (
                enriched_primary,
                enriched_secondary,
                enriched_tertiary,
                category_path,
                category_provenance,
            ) = combine_labels(osm_for_combination, provider_categories)
            opening_hours = (
                list(provider_result.opening_hours_text)
                if isinstance(provider_result.opening_hours_text, Sequence)
                and not isinstance(provider_result.opening_hours_text, (str, bytes))
                else None
            )
            props.update(
                {
                    "enriched_primary_label": enriched_primary,
                    "enriched_secondary_label": enriched_secondary,
                    "enriched_tertiary_label": enriched_tertiary,
                    "enriched_category_path": category_path,
                    "enriched_category_provenance": category_provenance,
                    "enriched_name": provider_result.name or "",
                    "enriched_name_provenance": provider_result.provider,
                    "enriched_rating": provider_result.rating,
                    "enriched_rating_provenance": provider_result.provider if provider_result.rating is not None else "",
                    "enriched_rating_count": provider_result.rating_count,
                    "enriched_opening_hours": opening_hours,
                    "enriched_opening_hours_provenance": provider_result.provider if opening_hours else "",
                    "provider_categories": provider_categories,
                    "provider_distance_m": provider_result.distance_m,
                    "provider_confidence": provider_result.confidence,
                    "provider_place_id": provider_result.place_id,
                    "provider_raw": provider_result.raw,
                }
            )
        else:
            (
                enriched_primary,
                enriched_secondary,
                enriched_tertiary,
                category_path,
                category_provenance,
            ) = combine_labels(osm_for_combination, [])
            props.update(
                {
                    "enriched_primary_label": enriched_primary,
                    "enriched_secondary_label": enriched_secondary,
                    "enriched_tertiary_label": enriched_tertiary,
                    "enriched_category_path": category_path,
                    "enriched_category_provenance": category_provenance,
                    "enriched_name": "",
                    "enriched_name_provenance": "",
                    "enriched_rating": None,
                    "enriched_rating_provenance": "",
                    "enriched_rating_count": None,
                    "enriched_opening_hours": None,
                    "enriched_opening_hours_provenance": "",
                    "provider_categories": [],
                    "provider_distance_m": None,
                    "provider_confidence": 0.0,
                    "provider_place_id": None,
                    "provider_raw": {},
                }
            )

        feature = dict(feature)
        feature["properties"] = props
        enriched.append(feature)

    summary = {
        "provider_name": provider.name,
        "provider_requests": provider_requests,
        "provider_matches": matched,
        "provider_match_rate": (matched / len(enriched)) if enriched else 0.0,
    }
    return enriched, summary


def run_generation(
    lat: float,
    lon: float,
    radius_m: float,
    resolution: float,
    output_dir: Path,
    overpass_url: str,
    fallback_overpass: Sequence[str],
    provider: ProviderBase,
    use_osm_labels: bool,
    provider_mode: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = compute_bbox(lat, lon, radius_m)
    projector = latlon_to_local_projector(lat, lon)

    requests_module = get_requests()
    osm_script.DEFAULT_OVERPASS_FALLBACKS = list(fallback_overpass)
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
    LOGGER.info("Rasterizing semantic map to %s", raster_output)
    rasterize_semantics(class_geometries, resolution, radius_m, raster_output)

    LOGGER.info("Building metadata GeoJSON with enrichment")
    features = build_building_features(buildings, projector, radius_m)
    enriched_features, provider_summary = enrich_features(features, provider, use_osm_labels)

    geojson = {
        "type": "FeatureCollection",
        "features": enriched_features,
    }
    buildings_path = output_dir / "buildings_enriched.geojson"
    with buildings_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    metadata = {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "resolution_m": resolution,
        "bbox": {
            "south": bbox[0],
            "west": bbox[1],
            "north": bbox[2],
            "east": bbox[3],
        },
        "class_to_id": CLASS_TO_ID,
        "semantic_map_path": str(raster_output),
        "semantic_map_preview": str(raster_output.with_suffix(".png")),
        "buildings_geojson": str(buildings_path),
        "overpass_url": overpass_url,
        "element_count": len(elements),
        "building_count": len(features),
        "enrichment": {
            "provider": provider_summary,
            "mode": provider_mode,
            "fields": [
                "enriched_primary_label",
                "enriched_secondary_label",
                "enriched_tertiary_label",
                "enriched_category_path",
                "enriched_name",
                "enriched_rating",
                "enriched_rating_count",
                "enriched_opening_hours",
            ],
            "use_osm_labels": use_osm_labels,
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    LOGGER.info("Generation complete: %s", metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate enriched semantic map and building taxonomy from OSM plus external providers.",
    )
    parser.add_argument("lat", type=float, help="Center latitude in decimal degrees")
    parser.add_argument("lon", type=float, help="Center longitude in decimal degrees")
    parser.add_argument(
        "radius",
        type=float,
        help="Half side length of the square region in meters",
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
        default=Path("./semantic_dataset_enriched"),
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--overpass-url",
        default="https://overpass-api.de/api/interpreter",
        help="Primary Overpass endpoint to query",
    )
    parser.add_argument(
        "--fallback-overpass",
        nargs="*",
        default=DEFAULT_OVERPASS_FALLBACKS,
        help="Optional list of fallback Overpass endpoints",
    )
    parser.add_argument(
        "--provider",
        choices=["osm", "google", "osm_google"],
        default="osm_google",
        help="Enrichment provider selection. 'osm' keeps OSM-only attributes, 'google' relies solely on Google metadata, and 'osm_google' merges both.",
    )
    parser.add_argument(
        "--google-api-key",
        default=None,
        help="Google Maps Platform API key for Google provider modes",
    )
    parser.add_argument(
        "--provider-radius",
        type=float,
        default=50.0,
        help="Search radius in meters for provider lookups",
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=35.0,
        help="Maximum centroid distance in meters to accept a provider match",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.2,
        help="Seconds to wait between provider requests to respect rate limits",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.provider in {"google", "osm_google"} and not args.google_api_key:
        args.google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    provider = select_provider(args)
    use_osm_labels = args.provider != "google"
    try:
        run_generation(
            lat=args.lat,
            lon=args.lon,
            radius_m=args.radius,
            resolution=args.resolution,
            output_dir=args.output,
            overpass_url=args.overpass_url,
            fallback_overpass=args.fallback_overpass,
            provider=provider,
            use_osm_labels=use_osm_labels,
            provider_mode=args.provider,
        )
    finally:
        provider.close()


if __name__ == "__main__":
    main()
