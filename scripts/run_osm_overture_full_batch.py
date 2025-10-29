#!/usr/bin/env python
"""Batch runner for OSM+Overture full workflow over multiple cities."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass
class City:
    name: str
    latitude: float
    longitude: float
    radius: Optional[float] = None
    resolution: Optional[float] = None


@dataclass
class CityResult:
    city: City
    status: str
    changed: bool
    message: str = ""


PER_CITY_TIMEOUT_S = int(os.getenv("PER_CITY_TIMEOUT_S", "1800"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for scripts/run_osm_overture_full.sh using a JSONL city list"
        )
    )
    parser.add_argument(
        "--cities-file",
        type=Path,
        default=Path("docs") / "cities" / "cities.jsonl",
        help="Path to the JSONL file containing city definitions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Directory where per-city outputs will be written.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2000.0,
        help="Radius in metres passed to the underlying script (default: 2000).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Raster resolution in metres (default: 0.5).",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Limit the number of cities processed from the top of the list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cities even if they were previously completed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel worker threads (default: 1)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry attempts per city after failure (default: 1)",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=5.0,
        help="Initial delay in seconds before retrying a failed city (exponential backoff)",
    )
    parser.add_argument(
        "--large-city-radius-threshold",
        type=float,
        default=4000.0,
        help="Radius in metres considered a 'large' city for concurrency throttling",
    )
    parser.add_argument(
        "--large-city-max-workers",
        type=int,
        default=2,
        help="Maximum concurrent workers allowed for large cities",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the batch runner (default: INFO)",
    )
    return parser.parse_args(argv)


def load_cities(path: Path, limit: Optional[int]) -> List[City]:
    if not path.exists():
        raise FileNotFoundError(f"City list not found: {path}")

    cities: List[City] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {err}"
                ) from err

            name = payload.get("name") or payload.get("city") or payload.get("title")
            lat = (
                payload.get("latitude")
                if payload.get("latitude") is not None
                else payload.get("lat")
            )
            lon = (
                payload.get("longitude")
                if payload.get("longitude") is not None
                else payload.get("lon")
            )
            radius = payload.get("radius") or payload.get("radius_m")
            resolution = payload.get("resolution") or payload.get("resolution_m")

            if name is None or lat is None or lon is None:
                raise ValueError(
                    "Each JSON object must contain 'name', 'latitude', and 'longitude' fields"
                )

            cities.append(
                City(
                    name=str(name),
                    latitude=float(lat),
                    longitude=float(lon),
                    radius=float(radius) if radius is not None else None,
                    resolution=float(resolution) if resolution is not None else None,
                )
            )

            if limit is not None and len(cities) >= limit:
                break
    if not cities:
        raise ValueError(f"No cities were loaded from {path}")
    return cities


def sanitize_city_name(name: str) -> str:
    sanitized = name.strip()
    sanitized = sanitized.replace("/", "-")
    sanitized = sanitized.replace("\\", "-")
    sanitized = sanitized.replace(":", "-")
    return sanitized


def city_completed_marker(city_dir: Path) -> Path:
    return city_dir / ".completed"


def run_city(
    run_script: Path,
    city: City,
    output_root: Path,
    radius: float,
    resolution: float,
    force: bool,
) -> bool:
    city_dir_name = sanitize_city_name(city.name)
    city_output_dir = output_root / city_dir_name
    city_output_dir.mkdir(parents=True, exist_ok=True)
    marker = city_completed_marker(city_output_dir)

    if marker.exists() and not force:
        return False

    if sys.platform == "win32":
        base_cmd = ["bash", str(run_script)]
    else:
        base_cmd = [str(run_script)]

    cmd = base_cmd + [
        f"{city.latitude}",
        f"{city.longitude}",
        f"{radius}",
        f"{resolution}",
        str(city_output_dir),
    ]

    subprocess.run(cmd, check=True, timeout=PER_CITY_TIMEOUT_S)

    marker.write_text("completed", encoding="utf-8")
    return True


def execute_city(
    run_script: Path,
    city: City,
    output_root: Path,
    radius: float,
    resolution: float,
    force: bool,
    retries: int,
    retry_base_delay: float,
    failed_path: Path,
    failed_lock: threading.Lock,
) -> CityResult:
    attempts = 0
    total_attempts = max(0, retries) + 1
    last_message = ""
    while attempts < total_attempts:
        try:
            changed = run_city(
                run_script=run_script,
                city=city,
                output_root=output_root,
                radius=radius,
                resolution=resolution,
                force=force,
            )
        except subprocess.TimeoutExpired:
            last_message = f"timeout after {PER_CITY_TIMEOUT_S}s"
            LOGGER.warning("City %s timed out after %ss", city.name, PER_CITY_TIMEOUT_S)
        except subprocess.CalledProcessError as err:
            last_message = f"exit code {err.returncode}"
            LOGGER.warning("City %s failed with exit code %s", city.name, err.returncode)
        else:
            status = "success" if changed else "skipped"
            return CityResult(city=city, status=status, changed=changed)
        attempts += 1
        if attempts < total_attempts:
            backoff = retry_base_delay * (2 ** (attempts - 1))
            jitter = random.uniform(0.7, 1.3)
            delay = backoff * jitter
            LOGGER.info(
                "Retrying city %s in %.1fs (attempt %d/%d)",
                city.name,
                delay,
                attempts + 1,
                total_attempts,
            )
            time.sleep(delay)
    with failed_lock:
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{city.name}\t{last_message or 'failed'}\n")
    return CityResult(city=city, status="failed", changed=False, message=last_message)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOGGER.debug("Batch arguments: %s", args)
    repo_root = Path(__file__).resolve().parents[1]
    run_script = repo_root / "scripts" / "run_osm_overture_full.sh"

    cities_file = (
        args.cities_file
        if args.cities_file.is_absolute()
        else repo_root / args.cities_file
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root / args.output_root
    )

    if not run_script.exists():
        raise FileNotFoundError(
            f"Unable to locate run_osm_overture_full.sh at {run_script}"
        )

    cities = load_cities(cities_file, args.max_cities)

    failed_path = output_root / "failed.txt"
    if failed_path.exists():
        failed_path.unlink()
    failed_lock = threading.Lock()
    large_city_threshold = max(0.0, args.large_city_radius_threshold)
    large_city_limit = max(1, args.large_city_max_workers)
    large_city_semaphore = threading.Semaphore(large_city_limit)
    max_workers = max(1, args.threads)
    retries = max(0, args.retries)
    retry_base_delay = max(0.0, args.retry_base_delay)

    def worker(city: City) -> CityResult:
        city_radius = city.radius if city.radius is not None else args.radius
        city_resolution = city.resolution if city.resolution is not None else args.resolution
        context = (
            large_city_semaphore
            if city_radius >= large_city_threshold
            else nullcontext()
        )
        with context:
            return execute_city(
                run_script=run_script,
                city=city,
                output_root=output_root,
                radius=city_radius,
                resolution=city_resolution,
                force=args.force,
                retries=retries,
                retry_base_delay=retry_base_delay,
                failed_path=failed_path,
                failed_lock=failed_lock,
            )

    processed = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(cities), unit="city", desc="Cities"
    ) as progress:
        future_map = {executor.submit(worker, city): city for city in cities}
        for future in as_completed(future_map):
            city = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected error while processing %s", city.name)
                with failed_lock:
                    failed_path.parent.mkdir(parents=True, exist_ok=True)
                    with failed_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"{city.name}\t{exc}\n")
                result = CityResult(city=city, status="failed", changed=False, message=str(exc))
            if result.status == "success":
                processed += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                failed += 1
                tqdm.write(f"City {result.city.name} failed: {result.message or 'see logs'}")
            progress.update(1)
            progress.set_postfix(processed=processed, skipped=skipped, failed=failed)

    if failed:
        LOGGER.warning(
            "Completed batch with %d failure(s). See %s for details.", failed, failed_path
        )
    else:
        LOGGER.info("Completed batch successfully: %d processed, %d skipped", processed, skipped)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
