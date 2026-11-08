#!/usr/bin/env bash
# G5: the stages after render, with the cycle check on Haiku (Batch) when the free Gemini quota is gone.
#   bash vi_math_verified_aug/arm_y/run_rest.sh <run_dir> <variant>
set -u
RUN="$1"; V="$2"; LOG="$RUN/scale_$V.log"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
P="E:/venv_tools/Scripts/python"
M="-m vi_math_verified_aug.arm_y.pipeline $RUN"
run() { echo "[scale] $* $(date +%T)" >> "$LOG"; $P $M "$@" >> "$LOG" 2>&1; }
run cycle "$V" haiku || exit 1
run gates || exit 1
run solve "$V" || exit 1
run vgate || exit 1
echo "[scale] rest $V done $(date +%T)" >> "$LOG"
