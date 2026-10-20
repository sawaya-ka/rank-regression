#!/usr/bin/env bash
set -euo pipefail
rank_run_dir=${1:?Run directory is required}
cd "$rank_run_dir"
exec python3 project/tools/gcp_pipeline.py _worker "$rank_run_dir"
