#!/bin/bash
# Run several experiment specs one after another, then compare them across maps; safe to re-run (resumes).
#
#   run_batch.sh OUT SPEC.yaml [SPEC.yaml ...]            # in the foreground
#   setsid nohup run_batch.sh OUT SPEC.yaml ... > /dev/null 2>&1 < /dev/null &   # detached: survives a closed
#                                                          # terminal or a lost session (not a container restart)
#
# Each spec runs into OUT/<spec file stem>/ with experiment.py run (every finished run is kept in runs/; running
# the same command again only does the missing ones). Then OUT/sweep_map.md / .png compare the specs by map.
# Progress: OUT/STATUS (running / done / failed, with the time), OUT/batch.log (everything), OUT/batch.pid.
# Stop it: kill $(cat OUT/batch.pid)  (then re-run the same command to resume).
set -u
OUT=$(realpath -m "$1"); shift
SRC=$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)
JOBS=${JOBS:-4}
mkdir -p "$OUT"
exec >> "$OUT/batch.log" 2>&1
echo $$ > "$OUT/batch.pid"
status() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S') $2" > "$OUT/STATUS"; echo "=== $1 $(date '+%H:%M:%S') $2"; }
cd "$SRC"
dirs=()
for spec in "$@"; do
  spec=$(realpath "$spec")
  name=$(basename "$spec" .yaml)
  dirs+=("$OUT/$name")
  status running "$name"
  if ! python3 mattbot_sim/scripts/experiment.py run "$spec" --out "$OUT/$name" --jobs "$JOBS" > "$OUT/$name.log" 2>&1; then
    status failed "$name (see $OUT/$name.log; re-run the same command to resume)"
    exit 1
  fi
  tail -1 "$OUT/$name.log"
done
if [ ${#dirs[@]} -gt 1 ]; then
  status running "sweep over maps"
  python3 mattbot_sim/scripts/experiment.py sweep "${dirs[@]}" --factor map --out "$OUT" > "$OUT/sweep_map.log" 2>&1 \
    || { status failed "sweep (see $OUT/sweep_map.log)"; exit 1; }
fi
status done "$(printf '%s ' "$@")"
