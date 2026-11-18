#!/usr/bin/env bash
# Wide-phase Teacher roll-in continuation and clean no-takeover evaluation.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
source_ckpt="${STUDENT_WIDE_SOURCE:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
root="${STUDENT_WIDE_ROOT:-$repo_dir/reports/php_student/wide_rollin_v1}"
mode="${1:-help}"

check() { [[ -f "$source_ckpt" && -x "$python" ]] || { echo '[ERROR] Missing checkpoint or Python' >&2; exit 2; }; }
activate() {
    source "$official_root/wbt_training/training_runs/setup_env.sh"
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    php_launcher_init; php_activate_sim
}
train() {
    [[ ! -e "$root/train" ]] || { echo "[ERROR] Refusing to overwrite $root/train" >&2; exit 2; }
    activate
    "$HSSIM_PYTHON" "$repo_dir/scripts/php_student/train_student_history_ab.py" \
      --checkpoint "$source_ckpt" --output-dir "$root/train" --arm single \
      --num-envs "${STUDENT_WIDE_NUM_ENVS:-256}" --iterations "${STUDENT_WIDE_ITERATIONS:-200}" \
      --learning-rate "${STUDENT_WIDE_LR:-0.0001}" --rollin-start 0.05 --rollin-end 0.85
}
eval_run() {
    [[ -s "$root/train/checkpoint.txt" ]] || { echo '[ERROR] Run train first' >&2; exit 2; }
    local ckpt; ckpt="$(<"$root/train/checkpoint.txt")"
    STUDENT_CHECKPOINT="$ckpt" STUDENT_EVAL_REPORT_DIR="$root/eval" \
    STUDENT_EVAL_EPISODES_PER_MOTION=3 STUDENT_EVAL_NUM_ENVS=40 STUDENT_EVAL_MAX_STEPS=6000 \
    STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 STUDENT_EVAL_TAKEOVER_MODE=none \
    STUDENT_EVAL_DIAGNOSTICS=1 PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}
case "$mode" in
  train) check; train ;;
  eval) check; eval_run ;;
  all) check; train; eval_run ;;
  *) echo "Usage: bash $0 train|eval|all"; [[ "$mode" == help ]] || exit 2 ;;
esac
