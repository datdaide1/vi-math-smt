#!/usr/bin/env bash
# G5 watchdog: run a resumable arm-Y stage; if the run directory's *.jsonl files stop growing for STALL seconds
# (e.g. a stalled 429 backoff), kill the job and resume it. Stops when the job exits 0 or after MAX restarts.
#   bash vi_math_verified_aug/arm_y/watchdog.sh <run_dir> <log> <python args...>
RUN="$1"; LOG="$2"; shift 2
STALL=${STALL:-300}; MAX=${MAX:-8}
size() { cat "$RUN"/*.jsonl 2>/dev/null | wc -c; }
for i in $(seq 1 "$MAX"); do
  echo "[watchdog] start #$i $(date +%T): $*" >> "$LOG"
  E:/venv_tools/Scripts/python "$@" >> "$LOG" 2>&1 &
  PID=$!
  last=$(size); t=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 15
    now=$(size)
    if [ "$now" != "$last" ]; then last=$now; t=0; else t=$((t + 15)); fi
    if [ "$t" -ge "$STALL" ]; then
      echo "[watchdog] no progress for ${STALL}s, killing $PID $(date +%T)" >> "$LOG"
      kill "$PID" 2>/dev/null; sleep 2; kill -9 "$PID" 2>/dev/null
      taskkill //F //PID "$PID" >/dev/null 2>&1
      break
    fi
  done
  wait "$PID" 2>/dev/null; rc=$?
  if [ "$t" -lt "$STALL" ] && [ "$rc" -eq 0 ]; then echo "[watchdog] done $(date +%T)" >> "$LOG"; exit 0; fi
  echo "[watchdog] exit rc=$rc, restarting" >> "$LOG"
done
echo "[watchdog] gave up after $MAX restarts" >> "$LOG"; exit 1
