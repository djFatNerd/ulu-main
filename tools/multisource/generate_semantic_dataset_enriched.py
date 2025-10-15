#!/usr/bin/env python3
"""Generate semantic datasets enriched with external provider metadata."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import re

from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FREE_GEOJSON = ROOT_DIR / "data" / "open" / "community_pois.geojson"
DEFAULT_FREE_CSV = ROOT_DIR / "data" / "open" / "business_registry.csv"
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
    local_to_latlon_projector,
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
    provenance: Dict[str, str]


class ProviderBase:
    """Base class for enrichment providers."""

    name = "base"

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - hook for future resources
        return None


def _normalize_tokens(value: Any) -> List[str]:
    """Normalize category- or tag-like values into a list of unique strings."""

    tokens: List[str] = []
    if value is None:
        return tokens
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.extend(_normalize_tokens(item))
        return tokens
    if isinstance(value, (int, float)):
        value = str(value)
    if isinstance(value, str):
        for part in re.split(r"[|;,]", value):
            cleaned = part.strip()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
    return tokens


def _get_nested_value(source: Dict[str, Any], field: Optional[str]) -> Any:
    """Retrieve nested dictionary values using dot-delimited field paths."""

    if not field:
        return None
    value: Any = source
    for part in field.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _extract_categories(source: Dict[str, Any], fields: Sequence[str]) -> List[str]:
    categories: List[str] = []
    for field in fields:
        value = _get_nested_value(source, field)
        for token in _normalize_tokens(value):
            if token not in categories:
                categories.append(token)
    return categories


def _extract_opening_hours(value: Any) -> Optional[List[str]]:
    hours = _normalize_tokens(value)
    return hours if hours else None


def _field_provenance(result: ProviderResult, field: str) -> str:
    return result.provenance.get(field) or (result.provider if result.provider else "")


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

        name_value = details.get("name") or best_candidate.get("name")
        provenance: Dict[str, str] = {}
        if place_id:
            provenance["place_id"] = self.name
        if name_value:
            provenance["name"] = self.name
        if categories:
            provenance["categories"] = self.name
        if rating is not None:
            provenance["rating"] = self.name
        if rating_count is not None:
            provenance["rating_count"] = self.name
        if opening_hours:
            provenance["opening_hours"] = self.name

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=name_value,
            categories=categories,
            rating=rating,
            rating_count=rating_count,
            opening_hours_text=opening_hours,
            raw={"candidate": best_candidate, "details": details},
            distance_m=best_distance,
            confidence=confidence,
            provenance=provenance,
        )


class OvertureBuildingsProvider(ProviderBase):
    """Fetch building metadata from the Overture Maps Places API."""

    name = "overture_buildings"

    def __init__(
        self,
        base_url: str,
        theme: str,
        search_radius_m: float,
        match_distance_m: float,
        sleep_between_requests: float,
        limit: int,
        include_fields: Sequence[str],
        category_fields: Sequence[str],
        name_fields: Sequence[str],
        auth_token: Optional[str],
        timeout: float,
        proxy_url: Optional[str],
    ) -> None:
        if not base_url:
            raise ValueError("Overture base URL cannot be empty")
        self._endpoint = base_url.rstrip("/")
        self._theme = theme
        self._search_radius_m = search_radius_m
        self._match_distance_m = match_distance_m
        self._sleep_between_requests = sleep_between_requests
        self._limit = max(1, limit)
        self._include_fields = [field for field in include_fields if field]
        self._category_fields = list(category_fields)
        self._name_fields = list(name_fields)
        self._auth_token = auth_token
        self._timeout = timeout
        self._requests = get_requests()
        self._session = self._requests.Session()
        self._session.trust_env = True
        self._session.verify = self._requests.certs.where()

        self._proxies: Dict[str, str] = {}
        if proxy_url:
            self._proxies = {"http": proxy_url, "https": proxy_url}
        else:
            env_proxies = self._requests.utils.get_environ_proxies(self._endpoint)
            # urllib3 expects lowercase scheme keys; filter empty entries.
            self._proxies = {
                scheme: value
                for scheme, value in env_proxies.items()
                if value
            }
        if self._proxies:
            self._session.proxies.update(self._proxies)

    def _throttle(self) -> None:
        if self._sleep_between_requests > 0:
            time.sleep(self._sleep_between_requests)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _bbox_from_radius(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        if self._search_radius_m <= 0:
            return (lon, lat, lon, lat)
        lat_buffer = self._search_radius_m / 111320.0
        lon_buffer = self._search_radius_m / max(1e-6, 111320.0 * math.cos(math.radians(lat)))
        south = lat - lat_buffer
        north = lat + lat_buffer
        west = lon - lon_buffer
        east = lon + lon_buffer
        return west, south, east, north

    def _extract_name(self, properties: Dict[str, Any]) -> Optional[str]:
        for field in self._name_fields:
            value = _get_nested_value(properties, field)
            if value is None:
                continue
            candidate: Optional[str] = None
            if isinstance(value, str):
                candidate = value.strip() or None
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        candidate = item.strip()
                        break
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("value")
                        if isinstance(text, str) and text.strip():
                            candidate = text.strip()
                            break
            elif isinstance(value, dict):
                text = value.get("text") or value.get("value") or value.get("name")
                if isinstance(text, str) and text.strip():
                    candidate = text.strip()
            if candidate:
                return candidate
        return None

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        params: Dict[str, Any] = {
            "theme": self._theme,
            "limit": self._limit,
        }
        west, south, east, north = self._bbox_from_radius(lat, lon)
        params["bbox"] = f"{west},{south},{east},{north}"
        if self._include_fields:
            params["include"] = ",".join(self._include_fields)

        try:
            self._throttle()
            response = self._session.get(
                self._endpoint,
                params=params,
                headers=self._headers(),
                timeout=self._timeout,
                proxies=self._proxies or None,
            )
            response.raise_for_status()
        except self._requests.exceptions.RequestException as exc:
            LOGGER.warning("Overture API request failed near %.6f, %.6f: %s", lat, lon, exc)
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            LOGGER.warning("Invalid JSON from Overture API near %.6f, %.6f: %s", lat, lon, exc)
            return None

        features = payload.get("features") if isinstance(payload, dict) else None
        if not features:
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        best_distance: Optional[float] = None

        for candidate in features:
            geometry = candidate.get("geometry") if isinstance(candidate, dict) else None
            centroid_lat: Optional[float] = None
            centroid_lon: Optional[float] = None
            if geometry:
                try:
                    geom = shape(geometry)
                    if not geom.is_empty:
                        centroid = geom.representative_point()
                        centroid_lat = centroid.y
                        centroid_lon = centroid.x
                except Exception:
                    centroid_lat = None
                    centroid_lon = None
            properties = candidate.get("properties") if isinstance(candidate, dict) else {}
            if (centroid_lat is None or centroid_lon is None) and isinstance(properties, dict):
                anchor = _get_nested_value(properties, "anchor")
                if isinstance(anchor, dict):
                    centroid_lat = anchor.get("lat") or anchor.get("latitude")
                    centroid_lon = anchor.get("lon") or anchor.get("lng") or anchor.get("longitude")

            distance = haversine_distance_m(lat, lon, centroid_lat, centroid_lon)
            if distance is None:
                continue
            if self._match_distance_m > 0 and distance > self._match_distance_m:
                continue
            if best_distance is None or distance < best_distance:
                best_candidate = candidate
                best_distance = distance

        if not best_candidate:
            return None

        properties = best_candidate.get("properties") if isinstance(best_candidate, dict) else {}
        properties = properties or {}
        name_value = self._extract_name(properties)
        categories = _extract_categories(properties, self._category_fields)
        if not categories:
            categories = ["building"]

        place_id = best_candidate.get("id") if isinstance(best_candidate, dict) else None
        if place_id is not None:
            place_id = str(place_id)

        confidence = 1.0
        if best_distance is not None and self._match_distance_m > 0:
            confidence = max(0.0, 1.0 - (best_distance / self._match_distance_m))

        provenance = {}
        if name_value:
            provenance["name"] = self.name
        if categories:
            provenance["categories"] = self.name
        if place_id:
            provenance["place_id"] = self.name

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=name_value,
            categories=categories,
            rating=None,
            rating_count=None,
            opening_hours_text=None,
            raw=best_candidate,
            distance_m=best_distance,
            confidence=confidence,
            provenance=provenance,
        )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass


class LocalGeoJSONProvider(ProviderBase):
    """Lookup enrichment metadata from an offline GeoJSON dataset."""

    name = "local_geojson"

    def __init__(
        self,
        path: Path,
        match_distance_m: float,
        name_field: Optional[str],
        category_fields: Sequence[str],
        id_field: Optional[str],
        rating_field: Optional[str],
        rating_count_field: Optional[str],
        opening_hours_field: Optional[str],
    ) -> None:
        self._path = path
        self._match_distance_m = match_distance_m
        self._name_field = name_field
        self._category_fields = list(category_fields)
        self._id_field = id_field
        self._rating_field = rating_field
        self._rating_count_field = rating_count_field
        self._opening_hours_field = opening_hours_field
        self._records: Dict[int, Dict[str, Any]] = {}
        self._geoms: List[Any] = []

        if not self._path.exists():
            raise FileNotFoundError(f"Local GeoJSON provider file not found: {self._path}")

        with self._path.open("r", encoding="utf-8") as f:
            content = json.load(f)

        features = content.get("features") if isinstance(content, dict) else None
        if features is None:
            raise ValueError("GeoJSON file must contain a FeatureCollection with a 'features' array")

        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty:
                continue
            props = feature.get("properties", {}) or {}
            centroid = geom.representative_point()
            place_id = props.get(self._id_field) if self._id_field else None
            if place_id is not None:
                place_id = str(place_id)
            record = {
                "centroid_lat": centroid.y,
                "centroid_lon": centroid.x,
                "properties": props,
                "name": props.get(self._name_field) if self._name_field else None,
                "categories": _extract_categories(props, self._category_fields),
                "place_id": place_id,
                "rating": props.get(self._rating_field) if self._rating_field else None,
                "rating_count": props.get(self._rating_count_field) if self._rating_count_field else None,
                "opening_hours": props.get(self._opening_hours_field) if self._opening_hours_field else None,
            }
            geom_id = id(geom)
            self._records[geom_id] = record
            self._geoms.append(geom)

        self._tree = STRtree(self._geoms) if self._geoms else None

    def _query_candidates(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        if not self._tree:
            return []
        if self._match_distance_m <= 0:
            envelope = box(lon, lat, lon, lat)
        else:
            lat_buffer = self._match_distance_m / 111320.0
            lon_buffer = self._match_distance_m / max(1e-6, 111320.0 * math.cos(math.radians(lat)))
            envelope = box(lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)
        candidates = []
        for geom in self._tree.query(envelope):
            record = self._records.get(id(geom))
            if record:
                candidates.append(record)
        return candidates

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        candidates = self._query_candidates(lat, lon)
        best_record: Optional[Dict[str, Any]] = None
        best_distance: Optional[float] = None
        for record in candidates:
            distance = haversine_distance_m(lat, lon, record["centroid_lat"], record["centroid_lon"])
            if distance is None:
                continue
            if self._match_distance_m > 0 and distance > self._match_distance_m:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_record = record

        if not best_record:
            return None

        props = best_record["properties"]
        name_value = best_record.get("name")
        categories = best_record.get("categories", [])
        place_id = best_record.get("place_id")

        rating = best_record.get("rating")
        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None

        rating_count = best_record.get("rating_count")
        try:
            rating_count = int(rating_count) if rating_count is not None else None
        except (TypeError, ValueError):
            rating_count = None

        opening_hours = _extract_opening_hours(best_record.get("opening_hours"))

        confidence = 0.0
        if best_distance is not None and self._match_distance_m > 0:
            confidence = max(0.0, 1.0 - (best_distance / self._match_distance_m))

        provenance: Dict[str, str] = {}
        if place_id:
            provenance["place_id"] = self.name
        if name_value:
            provenance["name"] = self.name
        if categories:
            provenance["categories"] = self.name
        if rating is not None:
            provenance["rating"] = self.name
        if rating_count is not None:
            provenance["rating_count"] = self.name
        if opening_hours:
            provenance["opening_hours"] = self.name

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=name_value,
            categories=categories,
            rating=rating,
            rating_count=rating_count,
            opening_hours_text=opening_hours,
            raw={"properties": props, "path": str(self._path)},
            distance_m=best_distance,
            confidence=confidence,
            provenance=provenance,
        )


class LocalCSVProvider(ProviderBase):
    """Lookup enrichment metadata from an offline CSV table."""

    name = "local_csv"

    def __init__(
        self,
        path: Path,
        match_distance_m: float,
        lat_field: str,
        lon_field: str,
        name_field: Optional[str],
        category_fields: Sequence[str],
        rating_field: Optional[str],
        rating_count_field: Optional[str],
        opening_hours_field: Optional[str],
        id_field: Optional[str],
    ) -> None:
        self._path = path
        self._match_distance_m = match_distance_m
        self._lat_field = lat_field
        self._lon_field = lon_field
        self._name_field = name_field
        self._category_fields = list(category_fields)
        self._rating_field = rating_field
        self._rating_count_field = rating_count_field
        self._opening_hours_field = opening_hours_field
        self._id_field = id_field
        self._records: Dict[int, Dict[str, Any]] = {}
        self._geoms: List[Any] = []

        if not self._path.exists():
            raise FileNotFoundError(f"Local CSV provider file not found: {self._path}")

        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get(self._lat_field))
                    lon = float(row.get(self._lon_field))
                except (TypeError, ValueError):
                    continue
                geom = Point(lon, lat)
                if geom.is_empty:
                    continue
                categories = _extract_categories(row, self._category_fields)
                place_id = row.get(self._id_field) if self._id_field else None
                if place_id is not None:
                    place_id = str(place_id)
                record = {
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "properties": row,
                    "name": row.get(self._name_field) if self._name_field else None,
                    "categories": categories,
                    "place_id": place_id,
                    "rating": row.get(self._rating_field) if self._rating_field else None,
                    "rating_count": row.get(self._rating_count_field) if self._rating_count_field else None,
                    "opening_hours": row.get(self._opening_hours_field) if self._opening_hours_field else None,
                }
                geom_id = id(geom)
                self._records[geom_id] = record
                self._geoms.append(geom)

        self._tree = STRtree(self._geoms) if self._geoms else None

    def _query_candidates(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        if not self._tree:
            return []
        if self._match_distance_m <= 0:
            envelope = box(lon, lat, lon, lat)
        else:
            lat_buffer = self._match_distance_m / 111320.0
            lon_buffer = self._match_distance_m / max(1e-6, 111320.0 * math.cos(math.radians(lat)))
            envelope = box(lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)
        candidates = []
        for geom in self._tree.query(envelope):
            record = self._records.get(id(geom))
            if record:
                candidates.append(record)
        return candidates

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        candidates = self._query_candidates(lat, lon)
        best_record: Optional[Dict[str, Any]] = None
        best_distance: Optional[float] = None
        for record in candidates:
            distance = haversine_distance_m(lat, lon, record["centroid_lat"], record["centroid_lon"])
            if distance is None:
                continue
            if self._match_distance_m > 0 and distance > self._match_distance_m:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_record = record

        if not best_record:
            return None

        name_value = best_record.get("name")
        categories = best_record.get("categories", [])

        rating = best_record.get("rating")
        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None

        rating_count = best_record.get("rating_count")
        try:
            rating_count = int(rating_count) if rating_count is not None else None
        except (TypeError, ValueError):
            rating_count = None

        opening_hours = _extract_opening_hours(best_record.get("opening_hours"))

        confidence = 0.0
        if best_distance is not None and self._match_distance_m > 0:
            confidence = max(0.0, 1.0 - (best_distance / self._match_distance_m))

        provenance: Dict[str, str] = {}
        if name_value:
            provenance["name"] = self.name
        if categories:
            provenance["categories"] = self.name
        if rating is not None:
            provenance["rating"] = self.name
        if rating_count is not None:
            provenance["rating_count"] = self.name
        if opening_hours:
            provenance["opening_hours"] = self.name

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=name_value,
            categories=categories,
            rating=rating,
            rating_count=rating_count,
            opening_hours_text=opening_hours,
            raw={"row": best_record["properties"], "path": str(self._path)},
            distance_m=best_distance,
            confidence=confidence,
            provenance=provenance,
        )


class CompositeProvider(ProviderBase):
    """Combine multiple providers and merge their responses."""

    def __init__(self, providers: Sequence[ProviderBase], label: Optional[str] = None) -> None:
        self._providers = list(providers)
        self.name = label or "+".join([p.name for p in self._providers]) or "null"

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        aggregated_categories: List[str] = []
        aggregated_raw: Dict[str, Any] = {}
        aggregated_provenance: Dict[str, str] = {}
        name_value: Optional[str] = None
        place_id: Optional[str] = None
        rating: Optional[float] = None
        rating_count: Optional[int] = None
        opening_hours: Optional[Sequence[str]] = None
        best_distance: Optional[float] = None
        best_confidence = 0.0
        matched = False

        for provider in self._providers:
            result = provider.lookup(lat, lon, feature)
            if not result:
                continue
            matched = True
            aggregated_raw[provider.name] = result.raw
            for field, source in result.provenance.items():
                aggregated_provenance.setdefault(field, source)
            if result.categories:
                for cat in result.categories:
                    if cat not in aggregated_categories:
                        aggregated_categories.append(cat)
            if name_value is None and result.name:
                name_value = result.name
            if place_id is None and result.place_id:
                place_id = result.place_id
            if rating is None and result.rating is not None:
                rating = result.rating
            if rating_count is None and result.rating_count is not None:
                rating_count = result.rating_count
            if opening_hours is None and result.opening_hours_text:
                opening_hours = result.opening_hours_text
            if result.distance_m is not None:
                if best_distance is None or result.distance_m < best_distance:
                    best_distance = result.distance_m
            best_confidence = max(best_confidence, result.confidence)

        if not matched:
            return None

        if aggregated_categories and "categories" not in aggregated_provenance:
            aggregated_provenance["categories"] = self.name
        if name_value and "name" not in aggregated_provenance:
            aggregated_provenance["name"] = self.name
        if place_id and "place_id" not in aggregated_provenance:
            aggregated_provenance["place_id"] = self.name
        if rating is not None and "rating" not in aggregated_provenance:
            aggregated_provenance["rating"] = self.name
        if rating_count is not None and "rating_count" not in aggregated_provenance:
            aggregated_provenance["rating_count"] = self.name
        if opening_hours and "opening_hours" not in aggregated_provenance:
            aggregated_provenance["opening_hours"] = self.name

        return ProviderResult(
            provider=self.name,
            place_id=place_id,
            name=name_value,
            categories=aggregated_categories,
            rating=rating,
            rating_count=rating_count,
            opening_hours_text=opening_hours,
            raw=aggregated_raw,
            distance_m=best_distance,
            confidence=best_confidence,
            provenance=aggregated_provenance,
        )

    def close(self) -> None:
        for provider in self._providers:
            provider.close()

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


def configure_free_tier(args: argparse.Namespace) -> None:
    """Mutate parsed arguments so only open/offline providers are used."""

    if not args.free_tier:
        return

    if args.providers:
        args.providers = [name for name in args.providers if name != "google"]
    if not args.providers:
        args.providers = []

    # Prefer user-specified overrides. Fall back to bundled open datasets when available.
    if not args.local_geojson:
        candidate_geojson = args.free_tier_geojson or DEFAULT_FREE_GEOJSON
        if candidate_geojson and Path(candidate_geojson).exists():
            args.local_geojson = Path(candidate_geojson)
        else:
            LOGGER.info("Free-tier GeoJSON source unavailable at %s", candidate_geojson)
    if not args.local_csv:
        candidate_csv = args.free_tier_csv or DEFAULT_FREE_CSV
        if candidate_csv and Path(candidate_csv).exists():
            args.local_csv = Path(candidate_csv)
        else:
            LOGGER.info("Free-tier CSV source unavailable at %s", candidate_csv)

    if args.local_geojson and "local_geojson" not in args.providers:
        args.providers.append("local_geojson")
    if args.local_csv and "local_csv" not in args.providers:
        args.providers.append("local_csv")

    if not args.providers:
        LOGGER.info(
            "Free-tier mode enabled but no offline providers configured; falling back to OSM-only labels."
        )

    # Ensure we do not rely on the Google shortcut defaults.
    args.provider = "osm"


def _append_once(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def select_provider(args: argparse.Namespace) -> Tuple[ProviderBase, bool, str]:
    use_osm_labels = not args.disable_osm_labels

    provider_names: List[str] = []
    explicit_providers = list(args.providers) if args.providers else []

    if explicit_providers:
        provider_names.extend(explicit_providers)
    else:
        if args.provider == "osm":
            provider = NullProvider()
            mode_parts = ["osm"] if use_osm_labels else []
            provider_mode = "+".join(mode_parts) if mode_parts else "none"
            return provider, use_osm_labels, provider_mode
        if args.provider == "google":
            provider_names = ["google"]
            if not args.disable_osm_labels:
                use_osm_labels = False
        elif args.provider == "osm_google":
            if args.google_api_key:
                provider_names = ["google"]
            else:
                LOGGER.info(
                    "Google API key not provided; continuing with OSM-only enrichment."
                )
                provider_names = []
        elif args.provider == "overture":
            provider_names = ["overture"]
            if not args.disable_osm_labels:
                use_osm_labels = False
        elif args.provider == "osm_overture":
            provider_names = ["overture"]
        else:
            raise ValueError(f"Unknown provider {args.provider}")

    if args.local_geojson:
        _append_once(provider_names, "local_geojson")
    if args.local_csv:
        _append_once(provider_names, "local_csv")

    provider_instances: List[ProviderBase] = []
    for name in provider_names:
        if name == "google":
            if not args.google_api_key:
                if explicit_providers or args.provider == "google":
                    raise ValueError(
                        "Google provider requested but no API key was supplied. Provide --google-api-key or set GOOGLE_MAPS_API_KEY."
                    )
                LOGGER.info("Skipping Google provider because no API key is configured.")
                continue
            provider_instances.append(
                GooglePlacesProvider(
                    api_key=args.google_api_key,
                    search_radius_m=args.provider_radius,
                    match_distance_m=args.match_distance,
                    sleep_between_requests=args.request_sleep,
                )
            )
        elif name == "overture":
            provider_instances.append(
                OvertureBuildingsProvider(
                    base_url=args.overture_endpoint,
                    theme=args.overture_theme,
                    search_radius_m=args.provider_radius,
                    match_distance_m=args.match_distance,
                    sleep_between_requests=args.request_sleep,
                    limit=args.overture_limit,
                    include_fields=args.overture_include_fields,
                    category_fields=args.overture_category_fields,
                    name_fields=args.overture_name_fields,
                    auth_token=args.overture_auth_token,
                    timeout=args.overture_timeout,
                    proxy_url=args.overture_proxy,
                )
            )
        elif name == "local_geojson":
            if not args.local_geojson:
                raise ValueError("--local-geojson must be provided when enabling local_geojson provider")
            provider_instances.append(
                LocalGeoJSONProvider(
                    path=args.local_geojson,
                    match_distance_m=args.match_distance,
                    name_field=args.local_geojson_name_field,
                    category_fields=args.local_geojson_category_fields,
                    id_field=args.local_geojson_id_field,
                    rating_field=args.local_geojson_rating_field,
                    rating_count_field=args.local_geojson_rating_count_field,
                    opening_hours_field=args.local_geojson_opening_hours_field,
                )
            )
        elif name == "local_csv":
            if not args.local_csv:
                raise ValueError("--local-csv must be provided when enabling local_csv provider")
            provider_instances.append(
                LocalCSVProvider(
                    path=args.local_csv,
                    match_distance_m=args.match_distance,
                    lat_field=args.local_csv_lat_field,
                    lon_field=args.local_csv_lon_field,
                    name_field=args.local_csv_name_field,
                    category_fields=args.local_csv_category_fields,
                    rating_field=args.local_csv_rating_field,
                    rating_count_field=args.local_csv_rating_count_field,
                    opening_hours_field=args.local_csv_opening_hours_field,
                    id_field=args.local_csv_id_field,
                )
            )
        else:
            raise ValueError(f"Unknown provider name '{name}' in --providers")

    if not provider_instances:
        provider = NullProvider()
        mode_parts = ["osm"] if use_osm_labels else []
        provider_mode = "+".join(mode_parts) if mode_parts else "none"
        return provider, use_osm_labels, provider_mode

    provider_label = "+".join([instance.name for instance in provider_instances])
    mode_parts = []
    if use_osm_labels:
        mode_parts.append("osm")
    if provider_label:
        mode_parts.append(provider_label)
    provider_mode = "+".join(mode_parts) if mode_parts else "none"

    return CompositeProvider(provider_instances, label=provider_label), use_osm_labels, provider_mode


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
            name_provenance = _field_provenance(provider_result, "name") if provider_result.name else ""
            rating_provenance = (
                _field_provenance(provider_result, "rating")
                if provider_result.rating is not None
                else ""
            )
            rating_count_provenance = (
                _field_provenance(provider_result, "rating_count")
                if provider_result.rating_count is not None
                else ""
            )
            opening_hours_provenance = (
                _field_provenance(provider_result, "opening_hours")
                if opening_hours
                else ""
            )
            place_id_provenance = (
                _field_provenance(provider_result, "place_id")
                if provider_result.place_id
                else ""
            )
            categories_provenance = (
                _field_provenance(provider_result, "categories")
                if provider_categories
                else ""
            )
            provider_sources = sorted({value for value in provider_result.provenance.values() if value})
            props.update(
                {
                    "enriched_primary_label": enriched_primary,
                    "enriched_secondary_label": enriched_secondary,
                    "enriched_tertiary_label": enriched_tertiary,
                    "enriched_category_path": category_path,
                    "enriched_category_provenance": category_provenance,
                    "enriched_name": provider_result.name or "",
                    "enriched_name_provenance": name_provenance,
                    "enriched_rating": provider_result.rating,
                    "enriched_rating_provenance": rating_provenance,
                    "enriched_rating_count": provider_result.rating_count,
                    "enriched_opening_hours": opening_hours,
                    "enriched_opening_hours_provenance": opening_hours_provenance,
                    "provider_categories": provider_categories,
                    "provider_categories_provenance": categories_provenance,
                    "provider_distance_m": provider_result.distance_m,
                    "provider_confidence": provider_result.confidence,
                    "provider_place_id": provider_result.place_id,
                    "provider_place_id_provenance": place_id_provenance,
                    "provider_raw": provider_result.raw,
                    "provider_sources": provider_sources,
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
                    "provider_categories_provenance": "",
                    "provider_distance_m": None,
                    "provider_confidence": 0.0,
                    "provider_place_id": None,
                    "provider_place_id_provenance": "",
                    "provider_raw": {},
                    "provider_sources": [],
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
    inverse_projector = local_to_latlon_projector(lat, lon)

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
    features = build_building_features(buildings, projector, inverse_projector, radius_m)
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
        choices=["osm", "google", "osm_google", "overture", "osm_overture"],
        default="osm_google",
        help=(
            "Backward-compatible provider shortcut. 'osm' keeps OSM-only attributes, "
            "'google' relies solely on Google metadata, 'overture' uses the Overture Maps "
            "buildings dataset, and 'osm_google'/'osm_overture' merge OSM labels with the "
            "respective provider. Use --providers for multi-source workflows."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["google", "overture", "local_geojson", "local_csv"],
        help="Optional list of enrichment providers to combine. Overrides --provider when supplied.",
    )
    parser.add_argument(
        "--free-tier",
        action="store_true",
        help=(
            "Enable an offline-friendly configuration that avoids paid providers. "
            "Automatically loads bundled open datasets unless overrides are supplied."
        ),
    )
    parser.add_argument(
        "--google-api-key",
        default=None,
        help="Google Maps Platform API key for Google provider modes",
    )
    parser.add_argument(
        "--overture-endpoint",
        default="https://api.overturemaps.org/places/v1/places",
        help="Overture Maps Places API endpoint used for building lookups",
    )
    parser.add_argument(
        "--overture-theme",
        default="buildings",
        help="Theme parameter passed to the Overture Places API",
    )
    parser.add_argument(
        "--overture-limit",
        type=int,
        default=25,
        help="Maximum number of features retrieved per Overture API request",
    )
    parser.add_argument(
        "--overture-include-fields",
        nargs="*",
        default=["names", "categories", "addresses"],
        help="Optional list of fields requested via the Overture include= parameter",
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
        "--overture-auth-token",
        default=None,
        help="Optional bearer token for Overture API access if required",
    )
    parser.add_argument(
        "--overture-timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for Overture API requests",
    )
    parser.add_argument(
        "--overture-proxy",
        default=None,
        help=(
            "Optional HTTP(S) proxy URL used for Overture requests. "
            "Defaults to the OVERTURE_PROXY environment variable when unset."
        ),
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
        "--disable-osm-labels",
        action="store_true",
        help="Disable combining OSM semantic labels with provider categories",
    )
    parser.add_argument(
        "--local-geojson",
        type=Path,
        default=None,
        help="Path to a local GeoJSON FeatureCollection used as an enrichment source",
    )
    parser.add_argument(
        "--local-geojson-name-field",
        default="name",
        help="GeoJSON property field used as the preferred name",
    )
    parser.add_argument(
        "--local-geojson-category-fields",
        nargs="+",
        default=["category", "type", "class", "primary", "secondary", "tertiary"],
        help="GeoJSON property fields that contain category labels",
    )
    parser.add_argument(
        "--local-geojson-id-field",
        default=None,
        help="GeoJSON property field that provides a stable identifier",
    )
    parser.add_argument(
        "--local-geojson-rating-field",
        default=None,
        help="GeoJSON property field that stores rating values",
    )
    parser.add_argument(
        "--local-geojson-rating-count-field",
        default=None,
        help="GeoJSON property field that stores rating counts",
    )
    parser.add_argument(
        "--local-geojson-opening-hours-field",
        default=None,
        help="GeoJSON property field that stores opening hours",
    )
    parser.add_argument(
        "--free-tier-geojson",
        type=Path,
        default=None,
        help="Override path for the GeoJSON dataset used when --free-tier is enabled",
    )
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=None,
        help="Path to a local CSV file used as an enrichment source",
    )
    parser.add_argument(
        "--local-csv-lat-field",
        default="lat",
        help="CSV column containing latitude values",
    )
    parser.add_argument(
        "--local-csv-lon-field",
        default="lon",
        help="CSV column containing longitude values",
    )
    parser.add_argument(
        "--local-csv-name-field",
        default="name",
        help="CSV column providing the preferred name",
    )
    parser.add_argument(
        "--local-csv-category-fields",
        nargs="+",
        default=["category", "type", "class"],
        help="CSV columns that contain category labels",
    )
    parser.add_argument(
        "--local-csv-id-field",
        default=None,
        help="CSV column that provides a stable identifier",
    )
    parser.add_argument(
        "--local-csv-rating-field",
        default=None,
        help="CSV column that stores rating values",
    )
    parser.add_argument(
        "--local-csv-rating-count-field",
        default=None,
        help="CSV column that stores rating counts",
    )
    parser.add_argument(
        "--local-csv-opening-hours-field",
        default=None,
        help="CSV column that stores opening hours",
    )
    parser.add_argument(
        "--free-tier-csv",
        type=Path,
        default=None,
        help="Override path for the CSV dataset used when --free-tier is enabled",
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

    if not args.google_api_key:
        args.google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not args.overture_auth_token:
        args.overture_auth_token = os.getenv("OVERTURE_AUTH_TOKEN")
    if not args.overture_proxy:
        args.overture_proxy = os.getenv("OVERTURE_PROXY")
    if args.overture_proxy:
        args.overture_proxy = args.overture_proxy.strip()
        if not args.overture_proxy:
            args.overture_proxy = None

    configure_free_tier(args)

    provider, use_osm_labels, provider_mode = select_provider(args)
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
            provider_mode=provider_mode,
        )
    finally:
        provider.close()


if __name__ == "__main__":
    main()
