#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE >&2
Usage: $0 LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
Example: $0 40.7580 -73.9855 1000 1.0 ./semantic_dataset_google_cost_down
USAGE
}

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage
  exit 1
fi

LAT="$1"
LON="$2"
RADIUS="$3"
RESOLUTION_VALUE="${4:-${RESOLUTION:-1.0}}"
OUTPUT_DIR="${5:-./semantic_dataset_google_cost_down}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
OVERPASS_URL_VALUE="${OVERPASS_URL:-https://overpass-api.de/api/interpreter}"
PROVIDERS_CSV="${GOOGLE_COST_DOWN_PROVIDERS:-osm,overture,google}"
MATCH_DISTANCE_VALUE="${MATCH_DISTANCE:-35.0}"
PROVIDER_RADIUS_VALUE="${PROVIDER_RADIUS:-50.0}"
PROVIDER_CACHE_QUANTIZATION_VALUE="${PROVIDER_CACHE_QUANTIZATION:-25.0}"
REQUEST_SLEEP_VALUE="${REQUEST_SLEEP:-0.2}"

GOOGLE_API_KEY="${GOOGLE_MAPS_API_KEY:-${GOOGLE_API_KEY:-}}"
if [[ "${PROVIDERS_CSV,,}" == *"google"* && -z "$GOOGLE_API_KEY" ]]; then
  echo "GOOGLE_MAPS_API_KEY (or GOOGLE_API_KEY) must be set when the Google provider is enabled." >&2
  exit 2
fi

CMD=(
  python tools/multisource/generate_semantic_dataset_cost_down.py
  "$LAT" "$LON" "$RADIUS"
  --resolution "$RESOLUTION_VALUE"
  --output "$OUTPUT_DIR"
  --overpass-url "$OVERPASS_URL_VALUE"
  --providers "$PROVIDERS_CSV"
  --match-distance "$MATCH_DISTANCE_VALUE"
  --provider-radius "$PROVIDER_RADIUS_VALUE"
  --provider-cache-quantization "$PROVIDER_CACHE_QUANTIZATION_VALUE"
  --request-sleep "$REQUEST_SLEEP_VALUE"
  --log-level "$LOG_LEVEL"
)

if [[ -n "$GOOGLE_API_KEY" ]]; then
  CMD+=(--google-api-key "$GOOGLE_API_KEY")
fi

if [[ -n "${FALLBACK_OVERPASS:-}" ]]; then
  IFS=' ' read -r -a FALLBACK_LIST <<< "${FALLBACK_OVERPASS}"
  CMD+=(--fallback-overpass "${FALLBACK_LIST[@]}")
fi

if [[ -n "${GOOGLE_MODE:-}" ]]; then
  CMD+=(--google-mode "${GOOGLE_MODE}")
fi

if [[ -n "${OVERTURE_ENDPOINT:-}" ]]; then
  CMD+=(--overture-endpoint "${OVERTURE_ENDPOINT}")
fi

if [[ -n "${OVERTURE_THEME:-}" ]]; then
  CMD+=(--overture-theme "${OVERTURE_THEME}")
fi

if [[ -n "${OVERTURE_LIMIT:-}" ]]; then
  CMD+=(--overture-limit "${OVERTURE_LIMIT}")
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
  CMD+=(--overture-timeout "${OVERTURE_TIMEOUT}")
fi

if [[ -n "${OVERTURE_CACHE_DIR:-}" ]]; then
  CMD+=(--overture-cache-dir "${OVERTURE_CACHE_DIR}")
fi

if [[ -n "${OVERTURE_CACHE_ONLY:-}" ]]; then
  case "${OVERTURE_CACHE_ONLY,,}" in
    1|true|yes|on)
      CMD+=(--overture-cache-only)
      ;;
  esac
fi

if [[ -n "${OVERTURE_PREFETCH_RADIUS:-}" ]]; then
  CMD+=(--overture-prefetch-radius "${OVERTURE_PREFETCH_RADIUS}")
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

exec "${CMD[@]}"
