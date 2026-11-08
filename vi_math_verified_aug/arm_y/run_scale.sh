#!/usr/bin/env bash
# G5 scale driver (VI then EN), cheapest-first, every stage resumable. Only the free Gemini cycle stage runs under the
# watchdog (a Batch wait has no store progress and must not be killed: a re-submit would pay twice).
#   bash vi_math_verified_aug/arm_y/run_scale.sh <run_dir> <variant: R2b|EN>
set -u
RUN="$1"; V="$2"; LOG="$RUN/scale_$V.log"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
P="E:/venv_tools/Scripts/python"
M="-m vi_math_verified_aug.arm_y.pipeline $RUN"
run() { echo "[scale] $* $(date +%T)" >> "$LOG"; $P $M "$@" >> "$LOG" 2>&1; }
if [ "$V" = "R2b" ]; then
  run formalize || exit 1
  run mutate || exit 1
fi
run render || exit 1
bash vi_math_verified_aug/arm_y/watchdog.sh "$RUN" "$LOG" $M cycle "$V" gemini || exit 1
run gates || exit 1
run solve "$V" || exit 1
run vgate || exit 1
echo "[scale] $V done $(date +%T)" >> "$LOG"
