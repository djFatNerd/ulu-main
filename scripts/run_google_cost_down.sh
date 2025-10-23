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
PROVIDERS_CSV="${GOOGLE_COST_DOWN_PROVIDERS:-osm,google}"

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
