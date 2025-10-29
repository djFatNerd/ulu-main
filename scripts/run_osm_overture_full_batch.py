#!/usr/bin/env python
"""Batch runner for OSM+Overture full workflow over multiple cities."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


@dataclass
class City:
    name: str
    latitude: float
    longitude: float


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

            if name is None or lat is None or lon is None:
                raise ValueError(
                    "Each JSON object must contain 'name', 'latitude', and 'longitude' fields"
                )

            cities.append(City(name=str(name), latitude=float(lat), longitude=float(lon)))

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

    cmd = [
        str(run_script),
        f"{city.latitude}",
        f"{city.longitude}",
        f"{radius}",
        f"{resolution}",
        str(city_output_dir),
    ]

    subprocess.run(cmd, check=True)

    marker.write_text("completed", encoding="utf-8")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
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

    processed = 0
    skipped = 0
    with tqdm(total=len(cities), unit="city", desc="Cities") as progress:
        for city in cities:
            try:
                changed = run_city(
                    run_script=run_script,
                    city=city,
                    output_root=output_root,
                    radius=args.radius,
                    resolution=args.resolution,
                    force=args.force,
                )
            except subprocess.CalledProcessError as err:
                tqdm.write(
                    f"Error while processing {city.name}: {err}. Resume will skip completed cities."
                )
                raise
            else:
                if changed:
                    processed += 1
                else:
                    skipped += 1
            progress.update(1)
            progress.set_postfix(processed=processed, skipped=skipped)

    return 0


if __name__ == "__main__":
    sys.exit(main())
