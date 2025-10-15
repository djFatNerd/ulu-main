#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE >&2
Usage: $0 LAT LON RADIUS [RESOLUTION] [OUTPUT_DIR]
Example: $0 40.7580 -73.9855 1000 1.0 ./semantic_dataset_osm_google
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
OUTPUT_DIR="${5:-./semantic_dataset_osm_google}"

MATCH_DISTANCE="${MATCH_DISTANCE:-35.0}"
PROVIDER_RADIUS="${PROVIDER_RADIUS:-50.0}"
REQUEST_SLEEP="${REQUEST_SLEEP:-0.2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

GOOGLE_API_KEY="${GOOGLE_MAPS_API_KEY:-${GOOGLE_API_KEY:-}}"
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "GOOGLE_MAPS_API_KEY (or GOOGLE_API_KEY) must be set for Google provider workflows." >&2
  exit 2
fi

CMD=(
  python tools/multisource/generate_semantic_dataset_enriched.py
  "$LAT" "$LON" "$RADIUS"
  --resolution "$RESOLUTION_VALUE"
  --output "$OUTPUT_DIR"
  --match-distance "$MATCH_DISTANCE"
  --provider-radius "$PROVIDER_RADIUS"
  --request-sleep "$REQUEST_SLEEP"
  --log-level "$LOG_LEVEL"
  --provider osm_google
  --google-api-key "$GOOGLE_API_KEY"
)

if [[ -n "${OVERPASS_URL:-}" ]]; then
  CMD+=(--overpass-url "$OVERPASS_URL")
fi

if [[ -n "${FALLBACK_OVERPASS:-}" ]]; then
  IFS=' ' read -r -a FALLBACK_LIST <<< "${FALLBACK_OVERPASS}"
  CMD+=(--fallback-overpass "${FALLBACK_LIST[@]}")
fi

exec "${CMD[@]}"
