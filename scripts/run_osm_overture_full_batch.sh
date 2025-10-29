#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_osm_overture_full_batch.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Unable to locate ${PYTHON_SCRIPT}" >&2
  exit 1
fi

exec python "${PYTHON_SCRIPT}" "$@"
