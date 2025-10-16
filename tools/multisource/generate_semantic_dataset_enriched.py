#!/usr/bin/env python3
"""Generate semantic datasets enriched with external provider metadata."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


@dataclass
class OvertureCliOptions:
    """Configuration detected for the overturemaps CLI interface."""

    bbox_flag: str = "--bbox"
    bbox_requires_separate: bool = False
    theme_flag: str = "--type"
    theme_uses_dataset_type: bool = True
    limit_flag: Optional[str] = None
    format_flag: str = "-f"
    available_flags: Set[str] = field(default_factory=set)


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
    """Fetch building metadata from cached Overture Maps Places downloads."""

    name = "overture_buildings"

    def __init__(
        self,
        base_url: str,
        theme: str,
        search_radius_m: float,
        cache_quantization_m: float,
        match_distance_m: float,
        sleep_between_requests: float,
        limit: int,
        include_fields: Sequence[str],
        category_fields: Sequence[str],
        name_fields: Sequence[str],
        timeout: float,
        cache_dir: Path,
        auth_token: Optional[str] = None,
        proxy: Optional[str] = None,
        cache_only: bool = False,
    ) -> None:
        del auth_token  # legacy parameter retained for CLI compatibility
        del proxy
        if importlib.util.find_spec("overturemaps") is None:
            raise RuntimeError(
                "overturemaps package is required for the Overture provider. Install it with 'pip install overturemaps'."
            )
        self._endpoint = base_url.rstrip("/") if base_url else "https://api.overturemaps.org/"
        self._theme = theme
        self._search_radius_m = search_radius_m
        self._cache_quantization_m = max(0.0, cache_quantization_m)
        self._match_distance_m = match_distance_m
        self._sleep_between_requests = sleep_between_requests
        self._limit = max(1, limit)
        self._include_fields = [field for field in include_fields if field]
        self._category_fields = list(category_fields)
        self._name_fields = list(name_fields)
        self._timeout = timeout
        self._cache_only = cache_only
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_type = self._derive_dataset_type(theme)
        self._cli_options = self._detect_cli_options()
        self._prefetch_bbox: Optional[Tuple[float, float, float, float]] = None
        self._prefetch_bbox_key: Optional[str] = None
        self._prefetch_bbox_value: Optional[str] = None

    def configure_prefetch_region(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: Optional[float],
    ) -> None:
        """Configure a global download bounding box shared across lookups."""

        if radius_m is None or radius_m <= 0:
            self._prefetch_bbox = None
            self._prefetch_bbox_key = None
            self._prefetch_bbox_value = None
            return

        south, west, north, east = compute_bbox(center_lat, center_lon, radius_m)
        west_south_east_north = (west, south, east, north)
        self._prefetch_bbox = west_south_east_north
        self._prefetch_bbox_key = f"{west:.6f}_{south:.6f}_{east:.6f}_{north:.6f}"
        self._prefetch_bbox_value = f"{west},{south},{east},{north}"
        if radius_m > self._search_radius_m:
            self._search_radius_m = radius_m

    def _detect_cli_options(self) -> OvertureCliOptions:
        """Inspect the overturemaps CLI to pick compatible flag names."""

        options = OvertureCliOptions()
        help_cmd = [sys.executable, "-m", "overturemaps.cli", "download", "--help"]
        try:
            result = subprocess.run(
                help_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            LOGGER.debug("Unable to introspect overturemaps CLI, using defaults: %s", exc)
            return options

        help_text = f"{result.stdout or ''}\n{result.stderr or ''}"
        flags = {
            match.lower()
            for match in re.findall(r"--[a-z0-9][a-z0-9_-]*", help_text, flags=re.IGNORECASE)
        }
        options.available_flags = flags

        for flag, uses_dataset in (
            ("--theme", False),
            ("--collection", False),
            ("--dataset", False),
            ("--type", True),
        ):
            if flag in flags:
                options.theme_flag = flag
                options.theme_uses_dataset_type = uses_dataset
                break

        for flag in ("--limit", "--max-results", "--maxresults"):
            if flag in flags:
                options.limit_flag = flag
                break

        for flag in ("--bbox", "--bounds"):
            if flag in flags:
                options.bbox_flag = flag
                break

        if "--format" in flags:
            options.format_flag = "--format"

        help_upper = help_text.upper()
        bbox_token = options.bbox_flag.upper()
        bbox_re = re.escape(bbox_token)
        if re.search(rf"{bbox_re}\s*(?:<[^>]+>\s*){4}", help_upper):
            options.bbox_requires_separate = True
        elif f"{bbox_token} FLOAT FLOAT FLOAT FLOAT" in help_upper or f"{bbox_token} NUMBER NUMBER NUMBER NUMBER" in help_upper:
            options.bbox_requires_separate = True
        elif f"{bbox_token} TEXT" in help_upper:
            options.bbox_requires_separate = False

        LOGGER.debug(
            "Detected overturemaps CLI flags: bbox=%s (separate=%s), theme=%s, limit=%s, format=%s",
            options.bbox_flag,
            options.bbox_requires_separate,
            options.theme_flag,
            options.limit_flag or "disabled",
            options.format_flag,
        )
        return options

    def _throttle(self) -> None:
        if self._sleep_between_requests > 0:
            time.sleep(self._sleep_between_requests)

    @staticmethod
    def _derive_dataset_type(theme: str) -> str:
        normalized = (theme or "").strip().lower()
        if not normalized:
            return "building"
        mapping = {
            "buildings": "building",
            "building": "building",
            "places": "place",
            "place": "place",
        }
        if normalized in mapping:
            return mapping[normalized]
        if normalized.endswith("s") and len(normalized) > 1:
            return normalized[:-1]
        return normalized

    def _bbox_from_radius(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        if self._prefetch_bbox is not None:
            return self._prefetch_bbox
        if self._search_radius_m <= 0:
            return (lon, lat, lon, lat)

        quantized_lat = lat
        quantized_lon = lon
        if self._cache_quantization_m > 0:
            lat_step = self._cache_quantization_m / 111320.0
            lon_step = self._cache_quantization_m / max(
                1e-6, 111320.0 * math.cos(math.radians(lat or 0.0))
            )
            if lat_step > 0:
                quantized_lat = round(lat / lat_step) * lat_step
            if lon_step > 0:
                quantized_lon = round(lon / lon_step) * lon_step

        lat_buffer = self._search_radius_m / 111320.0
        lon_buffer = self._search_radius_m / max(
            1e-6, 111320.0 * math.cos(math.radians(quantized_lat or 0.0))
        )
        south = quantized_lat - lat_buffer
        north = quantized_lat + lat_buffer
        west = quantized_lon - lon_buffer
        east = quantized_lon + lon_buffer
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

    def _cache_path(self, bbox_key: str) -> Path:
        include_hash = "defaults"
        if self._include_fields:
            include_hash = md5(",".join(sorted(self._include_fields)).encode("utf-8")).hexdigest()[:8]
        limit_part = f"limit{self._limit}"
        raw_key = f"{self._theme}_{limit_part}_{include_hash}_{bbox_key}"
        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_key)
        return self._cache_dir / f"{safe_key}.json"

    def _read_cached_payload(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Failed to read cached Overture data at %s: %s", cache_path, exc)
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None
        else:
            LOGGER.debug("Loaded cached Overture data from %s", cache_path)
            return payload

    def _build_cli_command_variants(self, bbox: str, tmp_path: Path) -> List[List[str]]:
        """Construct a set of overturemaps CLI invocations to try."""

        commands: List[List[str]] = []
        seen: Set[Tuple[str, ...]] = set()
        bbox_parts = [part.strip() for part in bbox.split(",") if part.strip()]
        available_flags = self._cli_options.available_flags

        def flag_allowed(flag: str) -> bool:
            return not available_flags or flag.lower() in available_flags

        theme_candidates: List[Tuple[str, str]] = []
        for flag, uses_dataset in [
            (self._cli_options.theme_flag, self._cli_options.theme_uses_dataset_type),
            ("--theme", False),
            ("--type", True),
            ("--dataset", False),
            ("--collection", False),
        ]:
            lower_flag = flag.lower()
            if any(existing_flag.lower() == lower_flag for existing_flag, _ in theme_candidates):
                continue
            if lower_flag == self._cli_options.theme_flag or flag_allowed(lower_flag):
                value = self._dataset_type if uses_dataset else self._theme
                theme_candidates.append((flag, value))
        if not theme_candidates:
            theme_candidates.append(("--type", self._dataset_type))

        limit_candidates: List[Optional[Tuple[str, str]]] = [None]
        if self._limit:
            limit_candidates = []
            candidate_flags = []
            if self._cli_options.limit_flag:
                candidate_flags.append(self._cli_options.limit_flag)
            candidate_flags.extend(["--limit", "--max-results"])
            seen_limit: Set[str] = set()
            for flag in candidate_flags:
                if not flag:
                    continue
                lower_flag = flag.lower()
                if lower_flag in seen_limit:
                    continue
                if lower_flag == (self._cli_options.limit_flag or "").lower() or flag_allowed(lower_flag):
                    limit_candidates.append((flag, str(self._limit)))
                    seen_limit.add(lower_flag)
            limit_candidates.append(None)

        bbox_flag_candidates: List[str] = []
        for flag in [self._cli_options.bbox_flag, "--bbox", "--bounds"]:
            lower_flag = flag.lower()
            if any(existing.lower() == lower_flag for existing in bbox_flag_candidates):
                continue
            if lower_flag == self._cli_options.bbox_flag or flag_allowed(lower_flag):
                bbox_flag_candidates.append(flag)
        if not bbox_flag_candidates:
            bbox_flag_candidates.append("--bbox")

        if self._cli_options.bbox_requires_separate:
            bbox_modes = ["separate", "comma", "equals"]
        else:
            bbox_modes = ["comma", "equals", "separate"]

        theme_modes = ["space", "equals"]
        format_flag = self._cli_options.format_flag

        for bbox_flag in bbox_flag_candidates:
            for bbox_mode in bbox_modes:
                for theme_flag, theme_value in theme_candidates:
                    for theme_mode in theme_modes:
                        for limit_option in limit_candidates:
                            cmd: List[str] = [
                                sys.executable,
                                "-m",
                                "overturemaps.cli",
                                "download",
                            ]
                            if bbox_mode == "equals":
                                cmd.append(f"{bbox_flag}={bbox}")
                            elif bbox_mode == "comma":
                                cmd.extend([bbox_flag, bbox])
                            else:
                                if len(bbox_parts) != 4:
                                    continue
                                cmd.append(bbox_flag)
                                cmd.extend(bbox_parts)
                            cmd.extend([format_flag, "geojson"])
                            if theme_mode == "equals":
                                cmd.append(f"{theme_flag}={theme_value}")
                            else:
                                cmd.extend([theme_flag, theme_value])
                            if limit_option:
                                limit_flag, limit_value = limit_option
                                cmd.extend([limit_flag, limit_value])
                            cmd.extend(["-o", str(tmp_path)])
                            key = tuple(cmd)
                            if key not in seen:
                                seen.add(key)
                                commands.append(cmd)

        if not commands:
            fallback_cmd = [
                sys.executable,
                "-m",
                "overturemaps.cli",
                "download",
                f"--bbox={bbox}",
                "-f",
                "geojson",
                "--type",
                self._dataset_type,
                "-o",
                str(tmp_path),
            ]
            if self._limit:
                fallback_cmd.extend(["--limit", str(self._limit)])
            commands.append(fallback_cmd)

        return commands

    def _download_payload(self, bbox: str, cache_path: Path) -> None:
        tmp_path = cache_path.with_suffix(".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        self._throttle()
        start = time.time()
        label = cache_path.name
        commands = self._build_cli_command_variants(bbox, tmp_path)
        timeout = max(self._timeout, 0) or None
        last_error: Optional[Tuple[subprocess.CalledProcessError, str]] = None
        attempt_count = len(commands)
        for attempt, cmd in enumerate(commands, start=1):
            LOGGER.debug(
                "Running overturemaps CLI (attempt %d/%d): %s",
                attempt,
                attempt_count,
                " ".join(cmd),
            )
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else ""
                stdout = exc.stdout.strip() if exc.stdout else ""
                message = "; ".join(part for part in [stdout, stderr] if part)
                last_error = (exc, message)
                fallbackable = (
                    exc.returncode == 2
                    or "usage" in message.lower()
                    or "no such option" in message.lower()
                    or "missing option" in message.lower()
                    or "invalid value" in message.lower()
                )
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                if fallbackable and attempt < attempt_count:
                    LOGGER.debug(
                        "Retrying overturemaps CLI with alternate arguments after failure: %s",
                        message or exc.returncode,
                    )
                    continue
                raise RuntimeError(
                    f"overturemaps CLI failed for bbox {bbox} (exit {exc.returncode}): {message}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"overturemaps CLI timed out for bbox {bbox} after {self._timeout} seconds"
                ) from exc
            else:
                duration = time.time() - start
                LOGGER.debug(
                    "Downloaded Overture data for %s in %.2f seconds via overturemaps CLI", label, duration
                )
                tmp_path.replace(cache_path)
                return

        if last_error is not None:
            exc, message = last_error
            raise RuntimeError(
                f"overturemaps CLI failed for bbox {bbox} (exit {exc.returncode}): {message}"
            ) from exc

        raise RuntimeError(
            f"overturemaps CLI failed for bbox {bbox}: unable to determine compatible arguments"
        )

    def _load_payload(
        self,
        lat: float,
        lon: float,
        bbox: str,
        cache_path: Path,
    ) -> Optional[Dict[str, Any]]:
        payload = self._read_cached_payload(cache_path)
        if payload is not None:
            return payload
        if self._cache_only:
            LOGGER.warning(
                "Overture cache miss for bbox %s near %.6f, %.6f (cache-only mode).",
                bbox,
                lat,
                lon,
            )
            LOGGER.info("Set OVERTURE_CACHE_DIR to a directory containing pre-downloaded responses.")
            return None
        try:
            LOGGER.info("Downloading Overture data for bbox %s via overturemaps", bbox)
            self._download_payload(bbox, cache_path)
        except Exception as exc:
            LOGGER.warning("Unexpected error downloading Overture data near %.6f, %.6f: %s", lat, lon, exc)
            return None
        payload = self._read_cached_payload(cache_path)
        if payload is None:
            LOGGER.warning(
                "Cached Overture data at %s could not be parsed after download near %.6f, %.6f",
                cache_path,
                lat,
                lon,
            )
        return payload

    def lookup(self, lat: float, lon: float, feature: Dict[str, Any]) -> Optional[ProviderResult]:
        west, south, east, north = self._bbox_from_radius(lat, lon)
        if self._prefetch_bbox_key is not None and self._prefetch_bbox_value is not None:
            bbox_key = self._prefetch_bbox_key
            bbox_value = self._prefetch_bbox_value
        else:
            bbox_key = f"{west:.6f}_{south:.6f}_{east:.6f}_{north:.6f}"
            bbox_value = f"{west},{south},{east},{north}"

        cache_path = self._cache_path(bbox_key)
        payload = self._load_payload(lat, lon, bbox_value, cache_path)
        if payload is None:
            return None

        features = payload.get("features") if isinstance(payload, dict) else None
        if not features:
            return None

        if self._limit and len(features) > self._limit:
            features = features[: self._limit]

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

        provenance: Dict[str, str] = {}
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
        if self._session is None:
            return
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
                cache_dir=args.overture_cache_dir,
                auth_token=args.overture_auth_token,
                proxy=args.overture_proxy,
                cache_only=args.overture_cache_only,
            )
            prefetch_radius = args.overture_prefetch_radius
            if prefetch_radius is None:
                prefetch_radius = args.radius
            overture_provider.configure_prefetch_region(args.lat, args.lon, prefetch_radius)
            provider_instances.append(overture_provider)
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
        help=(
            "Deprecated Overture API endpoint flag retained for backwards compatibility. "
            "Ignored when using the overturemaps CLI downloads."
        ),
    )
    parser.add_argument(
        "--overture-theme",
        default="buildings",
        help="Dataset theme passed to the overturemaps CLI (default: buildings)",
    )
    parser.add_argument(
        "--overture-limit",
        type=int,
        default=25,
        help="Maximum number of features retrieved per overturemaps CLI download",
    )
    parser.add_argument(
        "--overture-include-fields",
        nargs="*",
        default=["names", "categories", "addresses"],
        help="Optional list of fields to retain from downloaded Overture payloads (used for cache keying)",
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
        help="Timeout in seconds for overturemaps CLI invocations",
    )
    parser.add_argument(
        "--overture-auth-token",
        default=os.environ.get("OVERTURE_AUTH_TOKEN"),
        help=(
            "Deprecated Overture API token (default: OVERTURE_AUTH_TOKEN env variable). "
            "No longer required when using overturemaps downloads."
        ),
    )
    parser.add_argument(
        "--overture-proxy",
        default=os.environ.get("OVERTURE_PROXY"),
        help=(
            "Deprecated proxy flag retained for backwards compatibility. "
            "Ignored by the overturemaps CLI workflow."
        ),
    )
    parser.add_argument(
        "--overture-cache-dir",
        type=Path,
        default=None,
        help="Directory used to cache downloaded Overture datasets for offline reuse",
    )
    parser.add_argument(
        "--overture-cache-only",
        action="store_true",
        help=(
            "Skip Overture network requests and rely exclusively on cached responses. "
            "Use with OVERTURE_CACHE_DIR pointing at pre-downloaded payloads."
        ),
    )
    parser.add_argument(
        "--overture-prefetch-radius",
        type=float,
        default=None,
        help=(
            "Radius in meters for a shared Overture download bounding box. "
            "Defaults to the dataset radius when not provided."
        ),
    )
    parser.add_argument(
        "--provider-radius",
        type=float,
        default=50.0,
        help="Search radius in meters for provider lookups",
    )
    parser.add_argument(
        "--provider-cache-quantization",
        type=float,
        default=25.0,
        help=(
            "Quantization in meters applied to provider lookup coordinates before caching. "
            "Larger values increase cache reuse and reduce repeated downloads (default: 25.0)."
        ),
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
    cache_only_env = os.getenv("OVERTURE_CACHE_ONLY")
    if cache_only_env and not args.overture_cache_only:
        if cache_only_env.lower() in {"1", "true", "yes", "on"}:
            args.overture_cache_only = True
    if args.overture_cache_dir is None:
        env_cache_dir = os.getenv("OVERTURE_CACHE_DIR")
        if env_cache_dir:
            args.overture_cache_dir = Path(env_cache_dir).expanduser()
        else:
            args.overture_cache_dir = ROOT_DIR / "data" / "overture_cache"
    else:
        args.overture_cache_dir = Path(args.overture_cache_dir).expanduser()
    args.overture_cache_dir = args.overture_cache_dir.resolve()

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
