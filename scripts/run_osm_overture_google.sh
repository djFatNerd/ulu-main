#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE >&2
Usage: $0 [--cost-down|--google-mode MODE] LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
Example (full Google): $0 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google
Example (cost-down): $0 --cost-down 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_google_cost_down
USAGE
}

PIPELINE_MODE="full"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cost-down)
      PIPELINE_MODE="cost-down"
      shift
      ;;
    --google-mode)
      if [[ $# -lt 2 ]]; then
        echo "--google-mode requires an argument" >&2
        usage
        exit 1
      fi
      PIPELINE_MODE="$2"
      shift 2
      ;;
    --google-mode=*)
      PIPELINE_MODE="${1#*=}"
      shift
      ;;
    --full)
      PIPELINE_MODE="full"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage
  exit 1
fi

LAT="$1"
LON="$2"
RADIUS="$3"
RESOLUTION_VALUE="${4:-${RESOLUTION:-1.0}}"

DEFAULT_OUTPUT_FULL="./semantic_dataset_osm_overture_google"
DEFAULT_OUTPUT_COST_DOWN="./semantic_dataset_osm_overture_google_cost_down"
if [[ "$PIPELINE_MODE" == "cost-down" ]]; then
  DEFAULT_OUTPUT="$DEFAULT_OUTPUT_COST_DOWN"
else
  DEFAULT_OUTPUT="$DEFAULT_OUTPUT_FULL"
fi
OUTPUT_DIR="${5:-$DEFAULT_OUTPUT}"

MATCH_DISTANCE_VALUE="${MATCH_DISTANCE:-35.0}"
PROVIDER_RADIUS_VALUE="${PROVIDER_RADIUS:-50.0}"
PROVIDER_CACHE_QUANTIZATION_VALUE="${PROVIDER_CACHE_QUANTIZATION:-25.0}"
REQUEST_SLEEP_VALUE="${REQUEST_SLEEP:-0.2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

COST_DOWN_PROVIDERS="${GOOGLE_COST_DOWN_PROVIDERS:-osm,overture,google}"
NEEDS_GOOGLE=true
if [[ "$PIPELINE_MODE" == "cost-down" ]]; then
  if [[ "${COST_DOWN_PROVIDERS,,}" != *"google"* ]]; then
    NEEDS_GOOGLE=false
  fi
fi

GOOGLE_API_KEY="${GOOGLE_MAPS_API_KEY:-${GOOGLE_API_KEY:-}}"
if [[ "$NEEDS_GOOGLE" == true && -z "$GOOGLE_API_KEY" ]]; then
  echo "GOOGLE_MAPS_API_KEY (or GOOGLE_API_KEY) must be set for Google provider workflows." >&2
  exit 3
fi

if [[ "$PIPELINE_MODE" == "cost-down" ]]; then
  CMD=(
    python tools/multisource/generate_semantic_dataset_cost_down.py
    "$LAT" "$LON" "$RADIUS"
    --resolution "$RESOLUTION_VALUE"
    --output "$OUTPUT_DIR"
    --overpass-url "${OVERPASS_URL:-https://overpass-api.de/api/interpreter}"
    --providers "$COST_DOWN_PROVIDERS"
    --match-distance "$MATCH_DISTANCE_VALUE"
    --provider-radius "$PROVIDER_RADIUS_VALUE"
    --provider-cache-quantization "$PROVIDER_CACHE_QUANTIZATION_VALUE"
    --request-sleep "$REQUEST_SLEEP_VALUE"
    --log-level "$LOG_LEVEL"
    --google-api-key "$GOOGLE_API_KEY"
  )

  if [[ -n "${FALLBACK_OVERPASS:-}" ]]; then
    IFS=' ' read -r -a FALLBACK_LIST <<< "${FALLBACK_OVERPASS}"
    CMD+=(--fallback-overpass "${FALLBACK_LIST[@]}")
  fi

  if [[ -n "${GOOGLE_MODE:-}" ]]; then
    CMD+=(--google-mode "${GOOGLE_MODE}")
  fi

  if [[ -n "${GOOGLE_FIELDS:-}" ]]; then
    CMD+=(--google-fields "${GOOGLE_FIELDS}")
  fi

  if [[ -n "${GOOGLE_DISABLE_PHOTOS:-}" ]]; then
    CMD+=(--google-disable-photos "${GOOGLE_DISABLE_PHOTOS}")
  fi

  if [[ -n "${FEATURE_FILTER:-}" ]]; then
    CMD+=(--feature-filter "${FEATURE_FILTER}")
  fi

  if [[ -n "${MIN_AREA_M2:-}" ]]; then
    CMD+=(--min-area-m2 "${MIN_AREA_M2}")
  fi

  if [[ -n "${GRID_SIZE_M:-}" ]]; then
    CMD+=(--grid-size-m "${GRID_SIZE_M}")
  fi

  if [[ -n "${PROPAGATE_RADIUS_M:-}" ]]; then
    CMD+=(--propagate-radius-m "${PROPAGATE_RADIUS_M}")
  fi

  if [[ -n "${GOOGLE_BUDGET_REQUESTS:-}" ]]; then
    CMD+=(--google-budget-requests "${GOOGLE_BUDGET_REQUESTS}")
  fi

  if [[ -n "${GOOGLE_QPS_MAX:-}" ]]; then
    CMD+=(--google-qps-max "${GOOGLE_QPS_MAX}")
  fi

  if [[ -n "${UNCERTAINTY_THRESHOLD:-}" ]]; then
    CMD+=(--uncertainty-threshold "${UNCERTAINTY_THRESHOLD}")
  fi

  if [[ -n "${MAX_DISTANCE_M:-}" ]]; then
    CMD+=(--max-distance-m "${MAX_DISTANCE_M}")
  fi

  if [[ -n "${UNIT_PRICE_DETAILS_MIN:-}" ]]; then
    CMD+=(--unit-price-details-min "${UNIT_PRICE_DETAILS_MIN}")
  fi

  if [[ -n "${GOOGLE_CACHE_DIR:-}" ]]; then
    CMD+=(--google-cache-dir "${GOOGLE_CACHE_DIR}")
  fi

  if [[ -n "${PLACE_ID_PROPERTY:-}" ]]; then
    CMD+=(--place-id-property "${PLACE_ID_PROPERTY}")
  fi
else
  CMD=(
    python tools/multisource/generate_semantic_dataset_enriched.py
    "$LAT" "$LON" "$RADIUS"
    --resolution "$RESOLUTION_VALUE"
    --output "$OUTPUT_DIR"
    --match-distance "$MATCH_DISTANCE_VALUE"
    --provider-radius "$PROVIDER_RADIUS_VALUE"
    --provider-cache-quantization "$PROVIDER_CACHE_QUANTIZATION_VALUE"
    --request-sleep "$REQUEST_SLEEP_VALUE"
    --log-level "$LOG_LEVEL"
    --provider osm
    --providers overture google
    --google-api-key "$GOOGLE_API_KEY"
  )

  if [[ -n "${OVERPASS_URL:-}" ]]; then
    CMD+=(--overpass-url "$OVERPASS_URL")
  fi

  if [[ -n "${FALLBACK_OVERPASS:-}" ]]; then
    IFS=' ' read -r -a FALLBACK_LIST <<< "${FALLBACK_OVERPASS}"
    CMD+=(--fallback-overpass "${FALLBACK_LIST[@]}")
  fi
fi

if [[ -n "${OVERTURE_ENDPOINT:-}" ]]; then
  CMD+=(--overture-endpoint "$OVERTURE_ENDPOINT")
fi

if [[ -n "${OVERTURE_THEME:-}" ]]; then
  CMD+=(--overture-theme "$OVERTURE_THEME")
fi

if [[ -n "${OVERTURE_LIMIT:-}" ]]; then
  CMD+=(--overture-limit "$OVERTURE_LIMIT")
fi

if [[ -n "${OVERTURE_INCLUDE_FIELDS:-}" ]]; then
  IFS=' ' read -r -a INCLUDE_FIELDS <<< "${OVERTURE_INCLUDE_FIELDS}"
  CMD+=(--overture-include-fields "${INCLUDE_FIELDS[@]}")
fi

if [[ -n "${OVERTURE_CATEGORY_FIELDS:-}" ]]; then
  IFS=' ' read -r -a CATEGORY_FIELDS <<< "${OVERTURE_CATEGORY_FIELDS}"
  CMD+=(--overture-category-fields "${CATEGORY_FIELDS[@]}")
fi

if [[ -n "${OVERTURE_NAME_FIELDS:-}" ]]; then
  IFS=' ' read -r -a NAME_FIELDS <<< "${OVERTURE_NAME_FIELDS}"
  CMD+=(--overture-name-fields "${NAME_FIELDS[@]}")
fi

if [[ -n "${OVERTURE_TIMEOUT:-}" ]]; then
  CMD+=(--overture-timeout "$OVERTURE_TIMEOUT")
fi

if [[ -n "${OVERTURE_CACHE_DIR:-}" ]]; then
  CMD+=(--overture-cache-dir "$OVERTURE_CACHE_DIR")
fi

if [[ -n "${OVERTURE_CACHE_ONLY:-}" ]]; then
  case "${OVERTURE_CACHE_ONLY,,}" in
    1|true|yes|on)
      CMD+=(--overture-cache-only)
      ;;
  esac
fi

if [[ -n "${OVERTURE_PREFETCH_RADIUS:-}" ]]; then
  CMD+=(--overture-prefetch-radius "$OVERTURE_PREFETCH_RADIUS")
fi

exec "${CMD[@]}"
