#!/usr/bin/env bash
# D1 watchdog (lead's rule, 2026-11-07 23:00): run a bulk job; if its heartbeat file is not refreshed for 300 s, kill
# the whole process tree and resume it (every stage is resumable). Exit when the job exits 0 (done) or 3 (budget stop).
#   bash vi_math_verified_aug/scale_gen/watchdog.sh <logfile> <python args...>
set -u
LOG="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HB="$ROOT/data/train/N/_work/logs/heartbeat"
PY="/e/venv_tools/Scripts/python"
STALE=300
export PYTHONIOENCODING=utf-8
cd "$ROOT"
for attempt in $(seq 1 30); do
  echo "[watchdog] $(date '+%F %T') start attempt $attempt: $*" >> "$LOG"
  date +%s > "$HB.start"
  "$PY" "$@" >> "$LOG" 2>&1 &
  pid=$!
  killed=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    hb=$(stat -c %Y "$HB" 2>/dev/null || cat "$HB.start")
    st=$(cat "$HB.start")
    [ "$hb" -lt "$st" ] && hb=$st
    if [ $((now - hb)) -gt $STALE ]; then
      echo "[watchdog] $(date '+%F %T') heartbeat stale $((now - hb))s -> kill + resume" >> "$LOG"
      wp=$(cat /proc/$pid/winpid 2>/dev/null)
      [ -n "$wp" ] && taskkill //F //T //PID "$wp" >/dev/null 2>&1
      kill -9 "$pid" 2>/dev/null
      killed=1
      break
    fi
  done
  wait "$pid"; rc=$?
  echo "[watchdog] $(date '+%F %T') exit rc=$rc killed=$killed" >> "$LOG"
  if [ $killed -eq 0 ] && { [ $rc -eq 0 ] || [ $rc -eq 3 ]; }; then exit $rc; fi
  sleep 10
done
exit 1
