#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
output="${STUDENT_PHASE_PROBE_OUTPUT:-$repo_dir/reports/php_student/train_distribution_phase_probe_seed42.json}"
num_envs="${STUDENT_PHASE_PROBE_NUM_ENVS:-128}"
steps="${STUDENT_PHASE_PROBE_STEPS:-500}"

[[ -f "$checkpoint" && -x "$python" ]] || { echo "[ERROR] Missing checkpoint or Python" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "[ERROR] Refusing to overwrite $output" >&2; exit 2; }
[[ "$num_envs" =~ ^[1-9][0-9]*$ && "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid size" >&2; exit 2; }
source "$official_root/wbt_training/training_runs/setup_env.sh"
export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
php_launcher_init
php_activate_sim
"$HSSIM_PYTHON" "$repo_dir/scripts/php_student/probe_student_train_distribution.py" \
    --checkpoint "$checkpoint" --output "$output" --num-envs "$num_envs" --steps "$steps"
