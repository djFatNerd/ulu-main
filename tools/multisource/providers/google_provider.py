"""Google Places Details Essentials client with caching and throttling."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from utils.cache import JSONCache

LOGGER = logging.getLogger(__name__)

DETAILS_ENDPOINT = "https://maps.googleapis.com/maps/api/place/details/json"


@dataclass
class GoogleDetailsResult:
    place_id: str
    name: Optional[str]
    types: Tuple[str, ...]
    vicinity: Optional[str]
    raw: Dict[str, Any]


class GooglePlacesDetailsClient:
    """Minimal Google Places Details client enforcing Essentials usage."""

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: Path,
        fields: Tuple[str, ...],
        qps_max: float,
        budget_requests: Optional[int],
        disable_photos: bool = True,
        unit_price_usd_per_k: float = 20.0,
    ) -> None:
        self._session = requests.Session()
        self._session.trust_env = False
        self._api_key = api_key
        self._fields = tuple(sorted(set(fields)))
        self._qps_max = max(0.1, qps_max)
        self._sleep_between = 1.0 / self._qps_max
        self._last_call_ts: Optional[float] = None
        self._budget = budget_requests
        self._disable_photos = disable_photos
        self._unit_price = unit_price_usd_per_k

        self._cache = JSONCache(cache_dir / "place_details")
        self._coord_cache = JSONCache(cache_dir / "location_index")

        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._skip_budget = 0
        self._unit_price = unit_price_usd_per_k

        self._forbidden_fields = {
            "opening_hours",
            "formatted_phone_number",
            "international_phone_number",
            "website",
            "rating",
            "user_ratings_total",
            "price_level",
            "reviews",
            "photos",
        }
        if self._disable_photos and any(field == "photos" for field in self._fields):
            raise ValueError("Photos field is not permitted when google_disable_photos is true")

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def skipped_budget(self) -> int:
        return self._skip_budget

    @property
    def unit_price(self) -> float:
        return self._unit_price

    def _wait_for_qps(self) -> None:
        if self._last_call_ts is None:
            return
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._sleep_between:
            time.sleep(self._sleep_between - elapsed)

    def _within_budget(self) -> bool:
        if self._budget is None:
            return True
        return self._request_count < self._budget

    def estimate_cost(self) -> float:
        return (self._request_count * self._unit_price) / 1000.0

    def get_cached_by_location(self, lat: float, lon: float) -> Optional[GoogleDetailsResult]:
        key = f"{round(lat, 5)}_{round(lon, 5)}_details_min"
        cached = self._coord_cache.get(key)
        if cached.hit and cached.value:
            place_id = cached.value.get("place_id")
            if place_id:
                return self.get_details(place_id, lat=lat, lon=lon)
        return None

    def get_details(
        self, place_id: str, *, lat: Optional[float] = None, lon: Optional[float] = None
    ) -> Optional[GoogleDetailsResult]:
        cache_result = self._cache.get(place_id)
        if cache_result.hit and cache_result.value:
            self._cache_hits += 1
            payload = cache_result.value
            return GoogleDetailsResult(
                place_id=place_id,
                name=payload.get("name"),
                types=tuple(payload.get("types") or ()),
                vicinity=payload.get("vicinity"),
                raw=payload,
            )

        if not self._within_budget():
            self._skip_budget += 1
            return None

        self._cache_misses += 1
        params = {
            "place_id": place_id,
            "fields": ",".join(self._fields),
            "key": self._api_key,
        }

        self._wait_for_qps()
        response = self._session.get(DETAILS_ENDPOINT, params=params, timeout=30)
        self._last_call_ts = time.time()
        self._request_count += 1
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status != "OK":
            LOGGER.warning("Google details request failed for %s: %s", place_id, status)
            return None
        result = payload.get("result") or {}
        for field in list(result.keys()):
            if field in self._forbidden_fields:
                result.pop(field, None)
        result["types"] = tuple(result.get("types") or ())
        self._cache.set(place_id, result)
        if lat is not None and lon is not None:
            key = f"{round(lat, 5)}_{round(lon, 5)}_details_min"
            self._coord_cache.set(key, {"place_id": place_id})

        return GoogleDetailsResult(
            place_id=place_id,
            name=result.get("name"),
            types=result.get("types", ()),
            vicinity=result.get("vicinity"),
            raw=result,
        )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:  # pragma: no cover
            return
