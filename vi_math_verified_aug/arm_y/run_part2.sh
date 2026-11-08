#!/usr/bin/env bash
# G5 part 2 driver (lead 04:15): EN Y for D5's needs_EN.Y seeds, then VI wave 2 (render -> solve -> screen; the
# Qwen cycle runs separately, last). One heavy stage at a time; every stage resumable.
set -u
RUN=data/train/Y/_work/vi_scale; LOG="$RUN/part2.log"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 ARMY_DIRECT=1 ARMY_CLAUDE_INFLIGHT=14
P="E:/venv_tools/Scripts/python"
M="-m vi_math_verified_aug.arm_y.pipeline $RUN"
run() { echo "[part2] $* $(date +%T)" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
run $P $M en_selected data/train/final/selection_vi_v1_plus_provisional.json || exit 1
run $P $M gates || exit 1
ARMY_SCREEN_EN=1 run $P $M vgate || exit 1
run $P $M mutate || exit 1
ARMY_RENDER_VARIANTS=R2b run $P $M render || exit 1
run $P $M solve R2b || exit 1
run $P $M gates || exit 1
run $P $M vgate || exit 1
run $P $M status
echo "[part2] done $(date +%T)" >> "$LOG"
