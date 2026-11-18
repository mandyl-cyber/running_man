#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
checkpoint="${STUDENT_CHECKPOINT:-/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt}"
output="${STUDENT_STORAGE_PROBE_OUTPUT:-$repo_dir/reports/php_student/storage_coverage_seed42_v1.json}"
num_envs="${STUDENT_STORAGE_PROBE_NUM_ENVS:-128}"
rollouts="${STUDENT_STORAGE_PROBE_ROLLOUTS:-3}"
event_traces="${STUDENT_STORAGE_PROBE_EVENT_TRACES:-}"

[[ -f "$checkpoint" && -x "$python" ]] || { echo "[ERROR] Missing checkpoint or Python" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "[ERROR] Refusing to overwrite $output" >&2; exit 2; }
[[ "$num_envs" =~ ^[1-9][0-9]*$ && "$rollouts" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid probe size" >&2; exit 2; }
source "$official_root/wbt_training/training_runs/setup_env.sh"
export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
php_launcher_init
php_activate_sim
event_args=()
if [[ -n "$event_traces" ]]; then
    [[ -d "$event_traces" ]] || { echo "[ERROR] Missing event traces: $event_traces" >&2; exit 2; }
    event_args=(--event-traces "$event_traces")
fi
"$HSSIM_PYTHON" "$repo_dir/scripts/php_student/probe_student_storage_coverage.py" \
    --checkpoint "$checkpoint" --output "$output" --num-envs "$num_envs" --rollouts "$rollouts" \
    "${event_args[@]}"
[[ -s "$output" ]] || { echo "[ERROR] Probe exited without report: $output" >&2; exit 1; }
