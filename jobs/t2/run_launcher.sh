#!/usr/bin/env bash
# Watchdog for the T2 launcher: restarts the (resume-safe) loop if it dies before 17:00; state = jobs/queue/t2_state.json
cd /e/MATH-REASONING-THESIS/vi-math-smt
export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
for i in $(seq 1 20); do
  .venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher loop --every 120 --max-prio 3 --backend modal \
      --platform modal_l40s --modal-cap-usd 6.9 --workspace-cap-usd 29.5 --canary-mode seed0 --stop-at 17:00 >> jobs/logs/T2/launcher_loop.out 2>&1
  rc=$?
  echo "[watchdog $(date +%H:%M:%S)] launcher exited rc=$rc (attempt $i)" >> jobs/logs/T2/launcher_loop.out
  [ "$rc" -eq 0 ] && break
  [ "$(date +%H%M)" -ge 1700 ] && break
  sleep 30
done
echo "[watchdog $(date +%H:%M:%S)] done" >> jobs/logs/T2/launcher_loop.out
