#!/usr/bin/env bash
# Rebuild every table and figure from the raw Big Data Cup files.
# Usage: bash code/run_all.sh <raw_data_dir> <out_dir>
set -euo pipefail
RAW=${1:-data/raw}; OUT=${2:-out}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
python "$HERE/build_entries.py"   "$RAW" "$OUT/entries.csv" 20
python "$HERE/failed_carries.py"  "$RAW" "$OUT/failed_carries.csv" 10 20
python "$HERE/analyze.py"         "$OUT/entries.csv" "$OUT" 1000
python "$HERE/analyze_final.py"   "$OUT/entries.csv" "$OUT/failed_carries.csv" "$OUT" 1000
