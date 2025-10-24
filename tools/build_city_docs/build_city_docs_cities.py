"""Generate city-level JSON descriptors from the Overture divisions dataset.

This script queries the public Overture Maps S3 bucket for locality features and
emits a JSON document per city.  It is designed to be resilient to schema
changes where certain fields occasionally flip between scalar and MAP types.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import duckdb

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_URI = (
    "s3://overturemaps-us-west-2/release/2024-02-14.0/"
    "theme=divisions/type=division/subtype=locality/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/city_docs")
DEFAULT_MIN_POPULATION = 50_000
DEFAULT_FETCH_BATCH = 10_000

PREFERRED_LANGUAGE_KEYS = ("primary", "en", "und", "default")


@dataclass
class CityDoc:
    """Representation of a single city output document."""

    id: str
    name: str
    country: Optional[str]
    latitude: float
    longitude: float
    radius_m: int
    population: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "lat": self.latitude,
            "lon": self.longitude,
            "radius_m": self.radius_m,
            "population": self.population,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-uri",
        default=DEFAULT_DATASET_URI,
        help=(
            "S3 or local URI pointing to Overture divisions parquet files. "
            "Supports wildcards (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-city JSON files will be written.",
    )
    parser.add_argument(
        "--min-population",
        type=int,
        default=DEFAULT_MIN_POPULATION,
        help=(
            "Minimum population to treat a locality as a city when 'local_type' "
            "is not explicitly 'city' (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of city documents to emit.",
    )
    parser.add_argument(
        "--fetch-batch",
        type=int,
        default=DEFAULT_FETCH_BATCH,
        help="Number of locality rows to fetch per DuckDB batch (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def configure_duckdb(conn: duckdb.DuckDBPyConnection) -> None:
    """Install extensions and configure DuckDB for anonymous S3 access."""

    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute("INSTALL spatial")
    conn.execute("LOAD spatial")

    conn.execute("SET s3_region='us-west-2'")
    conn.execute("SET s3_access_key_id=''")
    conn.execute("SET s3_secret_access_key=''")
    conn.execute("SET s3_session_token=''")
    conn.execute("SET s3_endpoint='s3.amazonaws.com'")
    conn.execute("SET s3_url_style='path'")
    conn.execute("SET s3_use_ssl=true")


def query_localities(
    conn: duckdb.DuckDBPyConnection,
    dataset_uri: str,
    fetch_batch: int,
) -> Iterator[Dict[str, Any]]:
    """Stream locality rows from the divisions dataset."""

    sql = f"""
        SELECT
            id,
            iso_country_code,
            population,
            local_type,
            names,
            ST_Y(ST_PointOnSurface(geometry)) AS lat,
            ST_X(ST_PointOnSurface(geometry)) AS lon
        FROM read_parquet('{dataset_uri}')
        WHERE subtype = 'locality'
    """

    try:
        cursor = conn.execute(sql)
    except duckdb.IOException as exc:
        raise RuntimeError(
            "DuckDB could not find any parquet files matching the dataset URI "
            f"{dataset_uri!r}. If you are using the public Overture bucket, "
            "double-check the release path and ensure outbound network access "
            "is available."
        ) from exc
    columns = [desc[0] for desc in cursor.description]

    while True:
        rows = cursor.fetchmany(fetch_batch)
        if not rows:
            break
        for row in rows:
            yield dict(zip(columns, row))


def extract_preferred_string(value: Any, *, lowercase: bool = False) -> Optional[str]:
    """Return the most suitable textual representation from nested structures."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, dict):
        for key in PREFERRED_LANGUAGE_KEYS:
            if key in value:
                text = extract_preferred_string(value[key], lowercase=lowercase)
                if text:
                    return text
        for nested in value.values():
            text = extract_preferred_string(nested, lowercase=lowercase)
            if text:
                return text
        return None
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            text = extract_preferred_string(item, lowercase=lowercase)
            if text:
                return text
        return None
    else:
        return None

    return text.lower() if lowercase else text


def parse_population(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def normalise_local_type(value: Any) -> Optional[str]:
    text = extract_preferred_string(value, lowercase=True)
    return text if text else None


def choose_city_name(names: Any, fallback_id: str) -> Optional[str]:
    name = extract_preferred_string(names)
    if name:
        return name
    safe_id = fallback_id.rsplit("/", maxsplit=1)[-1]
    return safe_id if safe_id else None


def estimate_radius(population: Optional[int]) -> int:
    if not population:
        return 8_000
    if population >= 5_000_000:
        return 45_000
    if population >= 1_000_000:
        return 30_000
    if population >= 500_000:
        return 20_000
    if population >= 200_000:
        return 15_000
    if population >= 100_000:
        return 12_000
    if population >= 50_000:
        return 10_000
    return 8_000


def is_city(local_type: Optional[str], population: Optional[int], min_population: int) -> bool:
    if local_type == "city":
        return True
    if population and population >= min_population:
        return True
    return False


def sanitise_identifier(value: str) -> str:
    slug = value.replace(":", "_").replace("/", "_")
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "city"


def write_city_doc(city: CityDoc, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitise_identifier(city.id)}.json"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as stream:
        json.dump(city.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def build_city_docs(args: argparse.Namespace) -> Tuple[int, int]:
    conn = duckdb.connect()
    configure_duckdb(conn)

    created = 0
    errors = 0

    try:
        for record in query_localities(conn, args.dataset_uri, args.fetch_batch):
            try:
                local_type = normalise_local_type(record.get("local_type"))
                population = parse_population(record.get("population"))
                if not is_city(local_type, population, args.min_population):
                    continue

                name = choose_city_name(record.get("names"), record.get("id", ""))
                if not name:
                    LOGGER.debug(
                        "Skipping locality %s because no usable name was found",
                        record.get("id"),
                    )
                    continue

                country = extract_preferred_string(
                    record.get("iso_country_code"), lowercase=False
                )
                latitude = record.get("lat")
                longitude = record.get("lon")

                if latitude is None or longitude is None:
                    LOGGER.debug(
                        "Skipping locality %s due to missing coordinates", record.get("id")
                    )
                    continue

                city = CityDoc(
                    id=record.get("id", sanitise_identifier(name)),
                    name=name,
                    country=country,
                    latitude=float(latitude),
                    longitude=float(longitude),
                    radius_m=estimate_radius(population),
                    population=population,
                )

                write_city_doc(city, args.output_dir)
                created += 1

                if args.limit and created >= args.limit:
                    break
            except Exception:  # pragma: no cover - defensive logging path
                errors += 1
                LOGGER.exception("Failed to process locality record: %s", record)
    finally:
        conn.close()

    return created, errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        created, errors = build_city_docs(args)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        LOGGER.debug("Dataset resolution failure", exc_info=True)
        return 1
    LOGGER.info("City-level docs complete: created=%s, errors=%s", created, errors)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
