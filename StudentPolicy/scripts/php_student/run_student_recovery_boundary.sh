#!/usr/bin/env bash
# Diagnose whether the frozen Teacher can recover from Student-induced states.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
trace_root="${STUDENT_BOUNDARY_TRACE_ROOT:-$repo_dir/reports/php_student/teacher_state_action_audit_seed43_all40}"
root="${STUDENT_BOUNDARY_ROOT:-$repo_dir/reports/php_student/recovery_boundary_v1}"
checkpoint="${STUDENT_BOUNDARY_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
mode="${1:-run}"

audit() {
    mkdir -p "$root"
    "$python" "$repo_dir/scripts/php_student/audit_student_failure_boundary.py" \
      --trace-dir "$trace_root/traces" \
      --output "$root/failure_boundary.json" \
      --start-map "$root/recovery_start_frames.json"
}

evaluate() {
    audit
    [[ ! -s "$root/eval/full_clip_seed43.json" ]] || {
        echo "[SKIP] Existing recovery eval"; return;
    }
    STUDENT_CHECKPOINT="$checkpoint" STUDENT_EVAL_REPORT_DIR="$root/eval" \
    STUDENT_EVAL_EPISODES_PER_MOTION=1 STUDENT_EVAL_NUM_ENVS=40 \
    STUDENT_EVAL_MAX_STEPS=6000 STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 \
    STUDENT_EVAL_DIAGNOSTICS=1 STUDENT_EVAL_TAKEOVER_MODE=mapped_tail \
    STUDENT_EVAL_TAKEOVER_START_MAP="$root/recovery_start_frames.json" \
    PHP_EVAL_PROFILE=clean \
      bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}

case "$mode" in
    audit) audit ;;
    run) evaluate ;;
    *) echo "Usage: bash $0 audit|run" >&2; exit 2 ;;
esac
