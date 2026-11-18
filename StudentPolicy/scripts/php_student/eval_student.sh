#!/usr/bin/env bash
# Bounded rollout evaluation of the trained PHP depth student.
# This is NOT a per-clip, frame-0 full-clip success-rate test.
set -euo pipefail

mode="${1:-smoke}"
case "$mode" in
    smoke)
        default_envs=4
        default_steps=300
        ;;
    rollout)
        default_envs=16
        default_steps=3000
        ;;
    *)
        echo "Usage: $0 smoke|rollout [additional official eval_student.sh arguments]" >&2
        exit 2
        ;;
esac
shift || true

official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
checkpoint="${STUDENT_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
bundle="${PHP_BUNDLE:-/home/sjtu/.cache/php-parkour/motions-low-climb-76}"
num_envs="${STUDENT_EVAL_NUM_ENVS:-$default_envs}"
max_steps="${STUDENT_EVAL_STEPS:-$default_steps}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"

[[ -f "$checkpoint" ]] || { echo "[ERROR] Missing student checkpoint: $checkpoint" >&2; exit 2; }
[[ -d "$bundle" ]] || { echo "[ERROR] Missing motion bundle: $bundle" >&2; exit 2; }
[[ -f "$official_root/wbt_training/training_runs/eval_student.sh" ]] || {
    echo "[ERROR] Missing official student evaluator under $official_root" >&2; exit 2;
}
[[ -x "$python" ]] || { echo "[ERROR] Missing PHP Python: $python" >&2; exit 2; }
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] STUDENT_EVAL_NUM_ENVS must be positive" >&2; exit 2; }
[[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] STUDENT_EVAL_STEPS must be positive" >&2; exit 2; }

echo "[STUDENT EVAL] mode=$mode checkpoint=$checkpoint"
echo "[STUDENT EVAL] envs=$num_envs max_steps=$max_steps bundle=$bundle"
echo "[NOTE] This measures bounded rollout behavior; it does not establish per-clip full-clip success."

cd "$official_root"
CHECKPOINT="$checkpoint" REGISTRY="file://$bundle" \
NUM_ENVS="$num_envs" MAX_EVAL_STEPS="$max_steps" \
HSSIM_PYTHON="$python" LOGGER="${LOGGER:-disabled}" \
    bash wbt_training/training_runs/eval_student.sh \
        --training.headless True "$@"
