#!/usr/bin/env bash
set -euo pipefail
rank_project_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Prevent idle sleep on macOS while this client waits and downloads.
# Closing the lid or losing connectivity still requires --resume in direct mode.
if [[ $(uname -s) == Darwin ]] && command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i python3 "$rank_project_dir/tools/gcp_pipeline.py" "$@"
fi
exec python3 "$rank_project_dir/tools/gcp_pipeline.py" "$@"
