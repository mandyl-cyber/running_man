#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${STUDENT_PRECONTACT_AB_ROOT:-$repo_dir/reports/php_student/precontact_dagger_ab_v1}"
source_checkpoint="${STUDENT_PRECONTACT_SOURCE_CHECKPOINT:-$repo_dir/reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt}"
episodes="${STUDENT_PRECONTACT_EVAL_EPISODES:-3}"
seeds="${STUDENT_PRECONTACT_EVAL_SEEDS:-43 44 45 46}"

for arm in source baseline focus; do
    case "$arm" in
        source) checkpoint="$source_checkpoint" ;;
        *) checkpoint="$root/$arm/model_30197.pt" ;;
    esac
    [[ -f "$checkpoint" ]] || { echo "[ERROR] Missing $arm checkpoint: $checkpoint" >&2; exit 2; }
    for seed in $seeds; do
        [[ "$seed" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid seed: $seed" >&2; exit 2; }
        report_dir="$root/full_eval/$arm/seed$seed"
        report="$report_dir/full_clip_seed$seed.json"
        if [[ -s "$report" ]]; then
            echo "[SKIP] Existing report: $report"
            continue
        fi
        echo "[EVAL] arm=$arm seed=$seed episodes/clip=$episodes"
        STUDENT_CHECKPOINT="$checkpoint" \
        STUDENT_EVAL_EPISODES_PER_MOTION="$episodes" \
        STUDENT_EVAL_NUM_ENVS=40 \
        STUDENT_EVAL_MAX_STEPS=20000 \
        STUDENT_EVAL_SEED="$seed" \
        STUDENT_EVAL_OBSTACLE_SEED="$seed" \
        STUDENT_EVAL_REPORT_DIR="$report_dir" \
        STUDENT_EVAL_TAKEOVER_MODE=none \
        PHP_EVAL_PROFILE=clean \
        bash "$repo_dir/scripts/php_student/eval_student_full.sh"
    done
done

"${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}" \
    "$repo_dir/scripts/php_student/summarize_precontact_dagger_ab.py" \
    --root "$root" --episodes "$episodes" --seeds $seeds
