#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE >&2
Usage: $0 LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
Example: $0 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_overture_full
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
OUTPUT_DIR="${5:-./semantic_dataset_osm_overture_full}"

MATCH_DISTANCE="${MATCH_DISTANCE:-45.0}"
PROVIDER_RADIUS="${PROVIDER_RADIUS:-80.0}"
REQUEST_SLEEP="${REQUEST_SLEEP:-0.2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

CMD=(
  python tools/multisource/generate_semantic_dataset_osm_overture_full.py
  "$LAT" "$LON" "$RADIUS"
  --resolution "$RESOLUTION_VALUE"
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
  IFS=' ' read -r -a FALLBACK_LIST <<< "${FALLBACK_OVERPASS}"
  CMD+=(--fallback-overpass "${FALLBACK_LIST[@]}")
fi

if [[ -n "${FEATURE_FILTER:-}" ]]; then
  CMD+=(--feature-filter "$FEATURE_FILTER")
fi

if [[ -n "${MIN_AREA_M2:-}" ]]; then
  CMD+=(--min-area-m2 "$MIN_AREA_M2")
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
  CMD+=(--overture-cache-only)
fi

if [[ -n "${OVERTURE_PREFETCH_RADIUS:-}" ]]; then
  CMD+=(--overture-prefetch-radius "$OVERTURE_PREFETCH_RADIUS")
fi

exec "${CMD[@]}"

