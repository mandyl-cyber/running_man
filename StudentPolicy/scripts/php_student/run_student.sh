#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
teacher="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to model_19999.pt (or another approved PHP teacher)}"
bundle="${PHP_BUNDLE:-/home/sjtu/.cache/php-parkour/motions-low-climb-76}"
php_python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
mode="${1:-audit}"
seed="${PHP_STUDENT_SEED:-42}"
num_envs="${PHP_STUDENT_NUM_ENVS:-2048}"
iterations="${PHP_STUDENT_ITERATIONS:-30000}"
report_dir="$repo_dir/reports/php_student"
audit_report="$report_dir/teacher_bundle_audit.json"
contract_report="$report_dir/official_contract_audit.json"

[[ -x "$php_python" ]] || { echo "[ERROR] Missing PHP Python: $php_python" >&2; exit 2; }
[[ -x "$official_root/wbt_training/training_runs/run_terrain_warp_distill.sh" ]] || {
    echo "[ERROR] Missing official distillation launcher under: $official_root" >&2
    exit 2
}
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid PHP_STUDENT_NUM_ENVS" >&2; exit 2; }
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid PHP_STUDENT_ITERATIONS" >&2; exit 2; }

audit() {
    contract_args=(
        --official-root "$official_root"
        --output "$contract_report"
    )
    if [[ "${PHP_ALLOW_UNPINNED:-0}" == "1" ]]; then
        contract_args+=(--allow-unpinned)
    fi
    "$php_python" "$repo_dir/scripts/php_student/validate_official_contract.py" "${contract_args[@]}"
    "$php_python" "$repo_dir/scripts/php_student/inspect_teacher.py" \
        --checkpoint "$teacher" --bundle "$bundle" --output "$audit_report"
}

run_distill() {
    local run_envs="$1"
    local run_iterations="$2"
    local run_name="$3"
    audit
    echo "[STUDENT] repository=$repo_dir"
    echo "[STUDENT] backend=official_php_holosoma method=DAgger+PPO"
    echo "[STUDENT] teacher=$teacher"
    echo "[STUDENT] bundle=$bundle envs=$run_envs iterations=$run_iterations seed=$seed"
    cd "$official_root"
    TEACHER_CHECKPOINT="$teacher" \
    REGISTRY="file://$bundle" \
    FINETUNE=1 LOGGER="${LOGGER:-disabled}" NGPUS=1 \
        bash wbt_training/training_runs/run_terrain_warp_distill.sh \
            --training.headless True \
            --training.num-envs "$run_envs" \
            --training.seed "$seed" \
            --training.name "$run_name" \
            --algo.config.num-learning-iterations "$run_iterations" \
            --algo.config.save-interval "${PHP_STUDENT_SAVE_INTERVAL:-500}"
}

case "$mode" in
    audit)
        audit
        ;;
    preflight)
        run_distill "${PHP_PREFLIGHT_NUM_ENVS:-128}" "${PHP_PREFLIGHT_ITERATIONS:-5}" \
            "${RUN_NAME:-local-student-official-preflight-seed${seed}}"
        ;;
    train)
        run_distill "$num_envs" "$iterations" "${RUN_NAME:-local-student-official-seed${seed}}"
        ;;
    *)
        echo "Usage: $0 audit|preflight|train" >&2
        exit 2
        ;;
esac
