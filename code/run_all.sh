#!/usr/bin/env bash
# Rebuild every table and figure from the raw Big Data Cup files.
# Usage: bash code/run_all.sh <raw_data_dir> <out_dir>
# If <out_dir>/xg_bdc_predictions.csv exists (from xg_features.py + xg_train.py, which need the
# MoneyPuck files), the entries also get xG outcomes and analyze_xg.py runs (~5 min).
set -euo pipefail
RAW=${1:-data/raw}; OUT=${2:-out}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
XG="$OUT/xg_bdc_predictions.csv"
if [ -f "$XG" ]; then
  python "$HERE/build_entries.py" "$RAW" "$OUT/entries.csv" 20 "$XG"
else
  python "$HERE/build_entries.py" "$RAW" "$OUT/entries.csv" 20
fi
if [ -f "$XG" ]; then
  python "$HERE/failed_carries.py" "$RAW" "$OUT/failed_carries.csv" 10 20 "$XG"
else
  python "$HERE/failed_carries.py" "$RAW" "$OUT/failed_carries.csv" 10 20
fi
python "$HERE/analyze.py"         "$OUT/entries.csv" "$OUT" 1000
python "$HERE/analyze_final.py"   "$OUT/entries.csv" "$OUT/failed_carries.csv" "$OUT" 1000
if [ -f "$XG" ]; then
  python "$HERE/analyze_xg.py"    "$OUT/entries.csv" "$OUT/failed_carries.csv" "$OUT" 1000
fi
