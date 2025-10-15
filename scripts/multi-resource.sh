#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 LAT LON RADIUS [OUTPUT_DIR]" >&2
  echo "Example: $0 40.7580 -73.9855 1000 ./times_square_multi" >&2
  exit 1
fi

LAT="$1"
LON="$2"
RADIUS="$3"
OUTPUT_DIR="${4:-./semantic_dataset_multi}"

RESOLUTION="${RESOLUTION:-1.0}"
MATCH_DISTANCE="${MATCH_DISTANCE:-35.0}"
PROVIDER_RADIUS="${PROVIDER_RADIUS:-50.0}"
REQUEST_SLEEP="${REQUEST_SLEEP:-0.2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DEFAULT_PROVIDER_MODE="${DEFAULT_PROVIDER_MODE:-osm_google}"
FREE_TIER_MODE="${FREE_TIER:-false}"

GOOGLE_API_KEY="${GOOGLE_MAPS_API_KEY:-${GOOGLE_API_KEY:-}}"
GOOGLE_ENABLED=false
WANTS_GOOGLE=false
if [[ "$FREE_TIER_MODE" != "true" ]]; then
  case "$DEFAULT_PROVIDER_MODE" in
    google|osm_google)
      WANTS_GOOGLE=true
      ;;
  esac
  if [[ "${ENABLE_GOOGLE:-false}" == "true" ]]; then
    WANTS_GOOGLE=true
  fi
fi
if [[ "$WANTS_GOOGLE" == "true" ]]; then
  if [[ -n "$GOOGLE_API_KEY" ]]; then
    GOOGLE_ENABLED=true
    if [[ -z "${GOOGLE_MAPS_API_KEY:-}" ]]; then
      export GOOGLE_MAPS_API_KEY="$GOOGLE_API_KEY"
    fi
  elif [[ "${ENABLE_GOOGLE:-false}" == "true" ]]; then
    echo "ENABLE_GOOGLE requested but no GOOGLE_MAPS_API_KEY/GOOGLE_API_KEY provided" >&2
  fi
fi

CMD=(
  python tools/multisource/generate_semantic_dataset_enriched.py
  "$LAT" "$LON" "$RADIUS"
  --resolution "$RESOLUTION"
  --output "$OUTPUT_DIR"
  --match-distance "$MATCH_DISTANCE"
  --provider-radius "$PROVIDER_RADIUS"
  --request-sleep "$REQUEST_SLEEP"
  --log-level "$LOG_LEVEL"
)

if [[ -n "${OVERPASS_URL:-}" ]]; then
  CMD+=(--overpass-url "$OVERPASS_URL")
fi

if [[ -n "${FALLBACK_OVERPASS:-}" ]]; then
  # shellcheck disable=SC2206
  FALLBACK_VALUES=(${FALLBACK_OVERPASS})
  CMD+=(--fallback-overpass "${FALLBACK_VALUES[@]}")
fi

if [[ "${DISABLE_OSM_LABELS:-false}" == "true" ]]; then
  CMD+=(--disable-osm-labels)
fi

PROVIDERS=()
PROVIDER_ARGS=()

if [[ -n "${LOCAL_GEOJSON_PATH:-}" ]]; then
  if [[ ! -f "$LOCAL_GEOJSON_PATH" ]]; then
    echo "Local GeoJSON path $LOCAL_GEOJSON_PATH does not exist" >&2
    exit 2
  fi
  PROVIDERS+=(local_geojson)
  PROVIDER_ARGS+=(--local-geojson "$LOCAL_GEOJSON_PATH")
  if [[ -n "${LOCAL_GEOJSON_NAME_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-geojson-name-field "$LOCAL_GEOJSON_NAME_FIELD")
  fi
  if [[ -n "${LOCAL_GEOJSON_CATEGORY_FIELDS:-}" ]]; then
    # shellcheck disable=SC2206
    GEOJSON_FIELDS=(${LOCAL_GEOJSON_CATEGORY_FIELDS})
    PROVIDER_ARGS+=(--local-geojson-category-fields "${GEOJSON_FIELDS[@]}")
  fi
  if [[ -n "${LOCAL_GEOJSON_ID_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-geojson-id-field "$LOCAL_GEOJSON_ID_FIELD")
  fi
  if [[ -n "${LOCAL_GEOJSON_RATING_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-geojson-rating-field "$LOCAL_GEOJSON_RATING_FIELD")
  fi
  if [[ -n "${LOCAL_GEOJSON_RATING_COUNT_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-geojson-rating-count-field "$LOCAL_GEOJSON_RATING_COUNT_FIELD")
  fi
  if [[ -n "${LOCAL_GEOJSON_OPENING_HOURS_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-geojson-opening-hours-field "$LOCAL_GEOJSON_OPENING_HOURS_FIELD")
  fi
fi

if [[ -n "${LOCAL_CSV_PATH:-}" ]]; then
  if [[ ! -f "$LOCAL_CSV_PATH" ]]; then
    echo "Local CSV path $LOCAL_CSV_PATH does not exist" >&2
    exit 3
  fi
  PROVIDERS+=(local_csv)
  PROVIDER_ARGS+=(--local-csv "$LOCAL_CSV_PATH")
  if [[ -n "${LOCAL_CSV_LAT_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-lat-field "$LOCAL_CSV_LAT_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_LON_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-lon-field "$LOCAL_CSV_LON_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_NAME_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-name-field "$LOCAL_CSV_NAME_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_CATEGORY_FIELDS:-}" ]]; then
    # shellcheck disable=SC2206
    CSV_FIELDS=(${LOCAL_CSV_CATEGORY_FIELDS})
    PROVIDER_ARGS+=(--local-csv-category-fields "${CSV_FIELDS[@]}")
  fi
  if [[ -n "${LOCAL_CSV_ID_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-id-field "$LOCAL_CSV_ID_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_RATING_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-rating-field "$LOCAL_CSV_RATING_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_RATING_COUNT_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-rating-count-field "$LOCAL_CSV_RATING_COUNT_FIELD")
  fi
  if [[ -n "${LOCAL_CSV_OPENING_HOURS_FIELD:-}" ]]; then
    PROVIDER_ARGS+=(--local-csv-opening-hours-field "$LOCAL_CSV_OPENING_HOURS_FIELD")
  fi
fi

if [[ "$GOOGLE_ENABLED" == "true" ]]; then
  has_google=false
  for provider in "${PROVIDERS[@]}"; do
    if [[ "$provider" == "google" ]]; then
      has_google=true
      break
    fi
  done
  if [[ "$has_google" == "false" ]]; then
    PROVIDERS+=(google)
  fi
fi

if [[ ${#PROVIDERS[@]} -gt 0 ]]; then
  CMD+=(--providers "${PROVIDERS[@]}")
else
  CMD+=(--provider "$DEFAULT_PROVIDER_MODE")
fi

CMD+=("${PROVIDER_ARGS[@]}")

if [[ "$FREE_TIER_MODE" == "true" ]]; then
  CMD+=(--free-tier)
  if [[ -n "${FREE_TIER_GEOJSON:-}" ]]; then
    if [[ ! -f "$FREE_TIER_GEOJSON" ]]; then
      echo "Free-tier GeoJSON path $FREE_TIER_GEOJSON does not exist" >&2
      exit 4
    fi
    CMD+=(--free-tier-geojson "$FREE_TIER_GEOJSON")
  fi
  if [[ -n "${FREE_TIER_CSV:-}" ]]; then
    if [[ ! -f "$FREE_TIER_CSV" ]]; then
      echo "Free-tier CSV path $FREE_TIER_CSV does not exist" >&2
      exit 5
    fi
    CMD+=(--free-tier-csv "$FREE_TIER_CSV")
  fi
fi

if [[ "${VERBOSE:-false}" == "true" ]]; then
  echo "Running: ${CMD[*]}" >&2
fi

exec "${CMD[@]}"
