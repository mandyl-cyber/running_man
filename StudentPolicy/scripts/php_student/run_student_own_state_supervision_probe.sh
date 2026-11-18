#!/usr/bin/env bash
# Capture Student-induced states, then test whether Teacher actions are learnable offline.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_OWN_STATE_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
root="${STUDENT_OWN_STATE_ROOT:-$repo_dir/reports/php_student/own_state_supervision_probe_v1}"
mode="${1:-run}"

capture() {
  if [[ -s "$root/eval/full_clip_seed43.json" ]]; then echo "[SKIP] Existing Student trace eval"; return; fi
  STUDENT_CHECKPOINT="$checkpoint" STUDENT_EVAL_REPORT_DIR="$root/eval" \
  STUDENT_EVAL_EPISODES_PER_MOTION=1 STUDENT_EVAL_NUM_ENVS=40 STUDENT_EVAL_MAX_STEPS=6000 \
  STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 STUDENT_EVAL_DIAGNOSTICS=1 \
  STUDENT_EVAL_TRACE_MOTION_IDS=all STUDENT_EVAL_TRACE_DEPTH_STRIDE=1 \
  STUDENT_EVAL_TAKEOVER_MODE=none PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

probe() {
  local traces="$root/eval/traces"
  [[ -d "$traces" ]] || { echo "[ERROR] Missing Student-state traces: $traces" >&2; exit 2; }
  "$python" "$repo_dir/scripts/php_student/probe_observation_vs_state.py" \
    --checkpoint "$checkpoint" --teacher-traces "$traces" \
    --phase-start 0.25 --phase-end 0.65 --steps 600 --seed 42 \
    --output "$root/offline_student_state_probe.json"
}

case "$mode" in capture) capture ;; probe) probe ;; run) capture; probe ;; *) echo "Usage: $0 capture|probe|run" >&2; exit 2 ;; esac
