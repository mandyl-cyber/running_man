#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${STUDENT_PAIRED_ROLLOUT_ROOT:-$repo_dir/reports/php_student/continuation_collapse_v1/paired_rollout_v1}"
source_checkpoint="$repo_dir/reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt"
sweep_dir="$repo_dir/reports/php_student/continuation_collapse_v1/sweep/baseline"
seed="${STUDENT_PAIRED_ROLLOUT_SEED:-43}"
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid seed: $seed" >&2; exit 2; }

for arm in source step025 step050; do
    case "$arm" in
        source) checkpoint="$source_checkpoint" ;;
        step025) checkpoint="$sweep_dir/model_step_025.pt" ;;
        step050) checkpoint="$sweep_dir/model_step_050.pt" ;;
    esac
    [[ -f "$checkpoint" ]] || { echo "[ERROR] Missing checkpoint: $checkpoint" >&2; exit 2; }
    report_dir="$root/$arm"
    report="$report_dir/full_clip_seed$seed.json"
    if [[ -s "$report" ]]; then
        echo "[SKIP] Existing report: $report"
        continue
    fi
    echo "[PAIRED ROLLOUT] arm=$arm seed=$seed checkpoint=$checkpoint"
    STUDENT_CHECKPOINT="$checkpoint" \
    STUDENT_EVAL_EPISODES_PER_MOTION=1 \
    STUDENT_EVAL_NUM_ENVS=40 \
    STUDENT_EVAL_MAX_STEPS=20000 \
    STUDENT_EVAL_SEED="$seed" \
    STUDENT_EVAL_OBSTACLE_SEED="$seed" \
    STUDENT_EVAL_DIAGNOSTICS=1 \
    STUDENT_EVAL_TRACE_MOTION_IDS=all \
    STUDENT_EVAL_TRACE_DEPTH_STRIDE=5 \
    STUDENT_EVAL_TAKEOVER_MODE=none \
    STUDENT_EVAL_REPORT_DIR="$report_dir" \
    PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
done

if [[ -s "$root/paired_drift_summary.json" ]]; then
    echo "[SKIP] Existing paired summary: $root/paired_drift_summary.json"
else
    "${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}" \
        "$repo_dir/scripts/php_student/analyze_student_paired_rollout_drift.py" \
        --root "$root" --seed "$seed"
fi
