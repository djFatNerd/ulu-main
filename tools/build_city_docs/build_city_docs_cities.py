"""Generate city-level JSON descriptors from the Overture divisions dataset.

This script queries the public Overture Maps S3 bucket for locality features and
emits a JSON document per city.  It is designed to be resilient to schema
changes where certain fields occasionally flip between scalar and MAP types.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import duckdb

LOGGER = logging.getLogger(__name__)

# NOTE: The divisions dataset includes all subtypes within a single parquet
# collection.  We filter to localities within the query itself, so we only need
# to point DuckDB at the overarching divisions files.  Previously this constant
# attempted to concatenate two different globs, which produced an invalid URI
# such as ``.../*.parquets3://.../*.parquet``.  Keeping a single, valid glob
# ensures DuckDB can resolve the dataset successfully.
DEFAULT_DATASET_URI = (
    "s3://overturemaps-us-west-2/release/2024-02-14.0/"
    "theme=divisions/type=division/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/city_docs")
DEFAULT_MIN_POPULATION = 50_000
DEFAULT_FETCH_BATCH = 10_000

LOCAL_DATASET_DIR = Path("data/overture_sample")
LOCAL_DATASET_B64_PATH = LOCAL_DATASET_DIR / "divisions_sample.parquet.b64"
LOCAL_DATASET_FILENAME = "divisions_sample.parquet"


def ensure_local_sample_dataset() -> Optional[Path]:
    """Materialise the bundled sample dataset if the parquet file is missing."""

    parquet_path = LOCAL_DATASET_DIR / LOCAL_DATASET_FILENAME
    if parquet_path.exists():
        return parquet_path

    if not LOCAL_DATASET_B64_PATH.exists():
        LOGGER.warning(
            "Local sample dataset unavailable; expected to find %s.",
            LOCAL_DATASET_B64_PATH,
        )
        return None

    try:
        encoded = LOCAL_DATASET_B64_PATH.read_text()
        decoded = base64.b64decode(encoded)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Failed to decode bundled sample dataset from %s: %s",
            LOCAL_DATASET_B64_PATH,
            exc,
        )
        return None

    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_bytes(decoded)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOGGER.warning(
            "Unable to write local sample dataset to %s: %s",
            parquet_path,
            exc,
        )
        return None

    LOGGER.info("Materialised bundled local sample dataset at %s", parquet_path)
    return parquet_path


PREFERRED_LANGUAGE_KEYS = ("primary", "en", "und", "default")

# ``local_type`` values in Overture occasionally vary between releases.  The
# constants below capture the identifiers that clearly represent city-scale
# features so we can filter them without relying solely on population
# heuristics.
CITY_LOCAL_TYPE_VALUES = {
    "city",
    "capital",
    "capital_city",
    "megacity",
    "metropolis",
    "metropolitan_city",
    "municipality_city",
    "primary_city",
    "principal_city",
}

# Some localities include "city" in their identifier but represent smaller
# subdivisions.  Skip these explicitly so they do not bloat the global output.
CITY_LOCAL_TYPE_EXCLUDED = {"city_section", "city_district", "city_township"}

# A few releases emit namespaced identifiers such as "locality/city" or
# "locality:capital_city".  Treat these as city scale as well.
CITY_LOCAL_TYPE_SUFFIXES = ("/city", ":city", "_city")

# ``local_type`` identifiers that clearly represent non-city administrative
# divisions.  These values can legitimately have large populations, so relying
# solely on population thresholds would incorrectly classify them as cities.
# The identifiers are compared in lower case after normalisation and only apply
# when the value is not already covered by ``CITY_LOCAL_TYPE_VALUES``.
NON_CITY_LOCAL_TYPE_KEYWORDS = {
    "province",
    "state",
    "region",
    "county",
    "prefecture",
    "department",
    "territory",
    "governorate",
    "parish",
    "oblast",
    "canton",
    "division",
    "municipal_district",
    "island",
}


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


def configure_duckdb(conn: duckdb.DuckDBPyConnection) -> Tuple[bool, bool]:
    """Install extensions and configure DuckDB for anonymous S3 access.

    Returns:
        Tuple[bool, bool]: ``(spatial_available, httpfs_available)`` reflecting
        whether the optional extensions could be loaded. Offline environments
        may lack both extensions, in which case the script will rely on local
        parquet files that already include latitude/longitude columns.
    """

    httpfs_available = True
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except duckdb.Error as exc:  # pragma: no cover - depends on environment
        httpfs_available = False
        LOGGER.warning(
            "DuckDB httpfs extension unavailable; S3 access will be skipped. (%s)",
            exc,
        )

    spatial_available = True
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except duckdb.Error as exc:  # pragma: no cover - depends on environment
        spatial_available = False
        LOGGER.warning(
            "DuckDB spatial extension unavailable; falling back to datasets "
            "that include explicit latitude/longitude columns. (%s)",
            exc,
        )

    if httpfs_available:
        conn.execute("SET s3_region='us-west-2'")
        conn.execute("SET s3_access_key_id=''")
        conn.execute("SET s3_secret_access_key=''")
        conn.execute("SET s3_session_token=''")
        conn.execute("SET s3_endpoint='s3.amazonaws.com'")
        conn.execute("SET s3_url_style='path'")
        conn.execute("SET s3_use_ssl=true")

    return spatial_available, httpfs_available


def build_local_type_text_expression() -> str:
    """Return a DuckDB SQL expression that normalises ``local_type`` to text."""

    json_expr = "CAST(local_type AS JSON)"
    candidates = [
        f"TRY(lower(json_extract_string({json_expr}, '$.primary')))",
        f"TRY(lower(json_extract_string({json_expr}, '$.default')))",
        f"TRY(lower(json_extract_string({json_expr}, '$.en')))",
        f"TRY(lower(json_extract_string({json_expr}, '$.und')))",
        "TRY(lower(CAST(local_type AS VARCHAR)))",
    ]
    return "COALESCE(" + ", ".join(candidates + ["''"]) + ")"


def build_population_numeric_expression() -> str:
    """Return a DuckDB SQL expression that extracts a numeric population value."""

    json_expr = "CAST(population AS JSON)"
    candidates = [
        "TRY_CAST(population AS BIGINT)",
        f"TRY_CAST(json_extract({json_expr}, '$') AS BIGINT)",
        f"TRY_CAST(json_extract({json_expr}, '$.value') AS BIGINT)",
        f"TRY_CAST(json_extract({json_expr}, '$.total') AS BIGINT)",
        f"TRY_CAST(json_extract({json_expr}, '$.estimate') AS BIGINT)",
        f"TRY_CAST(json_extract({json_expr}, '$.population') AS BIGINT)",
    ]
    return "COALESCE(" + ", ".join(candidates) + ")"


def build_city_filter_sql(
    local_type_expr: str, population_expr: str, min_population: int
) -> str:
    """Return the SQL ``WHERE`` predicate that keeps city-scale localities only."""

    allowed_values = ", ".join(f"'{value}'" for value in sorted(CITY_LOCAL_TYPE_VALUES))
    excluded_values = ", ".join(f"'{value}'" for value in sorted(CITY_LOCAL_TYPE_EXCLUDED))

    suffix_conditions = [
        f"ends_with({local_type_expr}, '{suffix}')" for suffix in CITY_LOCAL_TYPE_SUFFIXES
    ]

    city_type_conditions = [f"{local_type_expr} IN ({allowed_values})"]
    if suffix_conditions:
        city_type_conditions.append("(" + " OR ".join(suffix_conditions) + ")")

    contains_condition = f"contains({local_type_expr}, 'city')"
    if excluded_values:
        contains_condition = (
            f"({contains_condition} AND {local_type_expr} NOT IN ({excluded_values}))"
        )
    city_type_conditions.extend(
        [
            contains_condition,
            f"starts_with({local_type_expr}, 'capital')",
        ]
    )

    type_filter = "(" + " OR ".join(city_type_conditions) + ")"

    population_filter = f"{population_expr} >= {min_population}"

    combined_filter = f"({type_filter} OR {population_filter})"

    if excluded_values:
        return f"({local_type_expr} NOT IN ({excluded_values}) AND {combined_filter})"
    return combined_filter


def resolve_dataset_uri(dataset_uri: str, *, httpfs_available: bool) -> str:
    """Resolve the dataset URI, considering environment overrides and fallbacks."""

    env_uri = os.environ.get("OVERTURE_DIVISIONS_DATASET_URI")
    if env_uri:
        LOGGER.info("Using dataset URI from $OVERTURE_DIVISIONS_DATASET_URI: %s", env_uri)
        return env_uri

    if dataset_uri == DEFAULT_DATASET_URI and not httpfs_available:
        local_dataset = ensure_local_sample_dataset()
        if local_dataset:
            local_uri = str(local_dataset)
            LOGGER.info(
                "DuckDB httpfs extension unavailable; falling back to bundled sample dataset: %s",
                local_uri,
            )
            return local_uri

    return dataset_uri


def determine_coordinate_expressions(
    available_columns: Sequence[str], spatial_available: bool
) -> Tuple[str, str]:
    """Return SQL expressions used to extract latitude and longitude."""

    column_lookup = {column.lower(): column for column in available_columns}

    if "geometry" in column_lookup:
        if not spatial_available:
            raise RuntimeError(
                "The dataset provides a geometry column but DuckDB's spatial "
                "extension is unavailable."
            )
        return (
            "ST_Y(ST_PointOnSurface(geometry))",
            "ST_X(ST_PointOnSurface(geometry))",
        )

    coordinate_pairs = [
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("y", "x"),
    ]
    for lat_key, lon_key in coordinate_pairs:
        if lat_key in column_lookup and lon_key in column_lookup:
            return (column_lookup[lat_key], column_lookup[lon_key])

    raise RuntimeError(
        "The dataset does not include geometry or explicit latitude/longitude columns."
    )


def query_localities(
    conn: duckdb.DuckDBPyConnection,
    dataset_uri: str,
    fetch_batch: int,
    spatial_available: bool,
    min_population: int,
) -> Iterator[Dict[str, Any]]:
    """Stream locality rows from the divisions dataset."""

    try:
        preview_cursor = conn.execute(
            f"SELECT * FROM read_parquet('{dataset_uri}') LIMIT 0"
        )
    except duckdb.IOException as exc:
        raise RuntimeError(
            "DuckDB could not find any parquet files matching the dataset URI "
            f"{dataset_uri!r}. If you are using the public Overture bucket, "
            "double-check the release path and ensure outbound network access "
            "is available."
        ) from exc

    column_names = [description[0] for description in preview_cursor.description]
    lat_expr, lon_expr = determine_coordinate_expressions(column_names, spatial_available)

    local_type_text_expr = build_local_type_text_expression()
    population_numeric_expr = build_population_numeric_expression()
    city_filter = build_city_filter_sql(
        local_type_text_expr, population_numeric_expr, min_population
    )

    sql = f"""
        SELECT
            id,
            iso_country_code,
            population,
            local_type,
            names,
            {lat_expr} AS lat,
            {lon_expr} AS lon,
            {local_type_text_expr} AS local_type_text,
            {population_numeric_expr} AS population_numeric
        FROM read_parquet('{dataset_uri}')
        WHERE subtype = 'locality'
          AND {city_filter}
    """

    cursor = conn.execute(sql)
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
        candidate = re.search(r"[-+]?\d[\d,._\s]*", value)
        if not candidate:
            return None
        cleaned = re.sub(r"[^0-9.+-]", "", candidate.group())
        try:
            numeric = float(cleaned)
        except ValueError:
            return None
        if math.isnan(numeric):
            return None
        suffix = value[candidate.end() :].lower()
        multiplier = 1
        if any(token in suffix for token in ("billion", "bn")):
            multiplier = 1_000_000_000
        elif any(token in suffix for token in ("million", "mn")):
            multiplier = 1_000_000
        elif any(token in suffix for token in ("thousand", "k")):
            multiplier = 1_000
        return int(numeric * multiplier)
    if isinstance(value, dict):
        for key in ("population", "value", "total", "estimate", "count"):
            if key in value:
                parsed = parse_population(value[key])
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = parse_population(nested)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            parsed = parse_population(item)
            if parsed is not None:
                return parsed
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
    if local_type:
        local_type = local_type.strip()
        if local_type in CITY_LOCAL_TYPE_VALUES:
            return True
        if local_type in CITY_LOCAL_TYPE_EXCLUDED:
            return False
        if any(local_type.endswith(suffix) for suffix in CITY_LOCAL_TYPE_SUFFIXES):
            return True
        if "city" in local_type:
            return True
        if local_type.startswith("capital"):
            return True
        # If the local type explicitly references a higher-level administrative
        # division (for example ``province`` or ``region``) treat it as a
        # non-city feature even when a large population is reported.
        for keyword in NON_CITY_LOCAL_TYPE_KEYWORDS:
            if keyword in local_type:
                return False
    return bool(population and population >= min_population)


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
    spatial_available, httpfs_available = configure_duckdb(conn)

    created = 0
    errors = 0

    dataset_uri = resolve_dataset_uri(args.dataset_uri, httpfs_available=httpfs_available)

    if dataset_uri.startswith("s3://") and not httpfs_available:
        raise RuntimeError(
            "DuckDB httpfs extension is unavailable, so S3 URIs cannot be read. "
            "Provide a local parquet dataset via --dataset-uri or set the "
            "$OVERTURE_DIVISIONS_DATASET_URI environment variable."
        )

    try:
        for record in query_localities(
            conn, dataset_uri, args.fetch_batch, spatial_available, args.min_population
        ):
            try:
                local_type = normalise_local_type(record.get("local_type"))
                population = parse_population(record.get("population"))
                if population is None:
                    population = parse_population(record.get("population_numeric"))
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
