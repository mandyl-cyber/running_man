#!/usr/bin/env bash
# Stage-gated DAgger curriculum. Stops and runs full eval on HOLD/REGRESS.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
official_root="${PHP_OFFICIAL_ROOT:-/home/sjtu/php_parkour_official}"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
profile="${STUDENT_CURRICULUM_PROFILE:-original}"
if [[ "$profile" == conservative ]]; then
    default_root="$repo_dir/reports/php_student/rollin_curriculum_v2_conservative"
    stage_names=(p090 p080 p070 p060 p050)
    stage_probabilities=(0.90 0.80 0.70 0.60 0.50)
    default_iterations=50
    default_lr=0.00003
elif [[ "$profile" == original ]]; then
    default_root="$repo_dir/reports/php_student/rollin_curriculum_v1"
    stage_names=(p075 p050 p025 p000)
    stage_probabilities=(0.75 0.50 0.25 0.00)
    default_iterations=150
    default_lr=0.0001
else
    echo "[ERROR] STUDENT_CURRICULUM_PROFILE must be original or conservative" >&2; exit 2
fi
root="${STUDENT_CURRICULUM_ROOT:-$default_root}"
initial="${STUDENT_CURRICULUM_SOURCE:-$repo_dir/reports/php_student/wide_rollin_v1/train/model_30198.pt}"
mode="${1:-help}"
quick_seeds=(43 44)

check() {
    [[ -f "$initial" && -x "$python" && -f "$official_root/wbt_training/training_runs/setup_env.sh" ]] || {
        echo "[ERROR] Missing source checkpoint, Python, or official setup" >&2; exit 2;
    }
}
activate() {
    source "$official_root/wbt_training/training_runs/setup_env.sh"
    export HSSIM_PYTHON="$python" LOGGER=disabled PYTHONUNBUFFERED=1
    php_launcher_init; php_activate_sim
}
eval_one() {
    local checkpoint="$1" out="$2" seed="$3" episodes="$4"
    local report="$out/full_clip_seed${seed}.json"
    if [[ -s "$report" ]]; then
        echo "[SKIP] Existing eval: $report"; return
    fi
    STUDENT_CHECKPOINT="$checkpoint" STUDENT_EVAL_REPORT_DIR="$out" \
    STUDENT_EVAL_EPISODES_PER_MOTION="$episodes" STUDENT_EVAL_NUM_ENVS=40 \
    STUDENT_EVAL_MAX_STEPS=6000 STUDENT_EVAL_SEED="$seed" STUDENT_EVAL_OBSTACLE_SEED="$seed" \
    STUDENT_EVAL_TAKEOVER_MODE=none STUDENT_EVAL_DIAGNOSTICS=1 PHP_EVAL_PROFILE=clean \
    bash "$repo_dir/scripts/php_student/eval_student_full.sh"
}
eval_pair() {
    local checkpoint="$1" out="$2" episodes="$3"
    for seed in "${quick_seeds[@]}"; do eval_one "$checkpoint" "$out/seed$seed" "$seed" "$episodes"; done
}
train_stage() {
    local source="$1" stage="$2" probability="$3" out="$root/stages/$stage/train"
    if [[ ! -s "$out/checkpoint.txt" ]]; then
        [[ ! -e "$out" ]] || { echo "[ERROR] Incomplete existing stage: $out" >&2; exit 2; }
        activate
        "$HSSIM_PYTHON" "$repo_dir/scripts/php_student/train_student_history_ab.py" \
          --checkpoint "$source" --output-dir "$out" --arm single \
          --num-envs "${STUDENT_CURRICULUM_NUM_ENVS:-256}" \
          --iterations "${STUDENT_CURRICULUM_ITERATIONS:-$default_iterations}" \
          --learning-rate "${STUDENT_CURRICULUM_LR:-$default_lr}" \
          --rollin-start 0.05 --rollin-end 0.85 --rollin-probability "$probability"
    else
        echo "[SKIP] Existing stage checkpoint: $out/checkpoint.txt"
    fi
}
full_eval_and_stop() {
    local incumbent="$1" candidate="$2" stage="$3" decision="$4"
    echo "[$decision] Running paired full evaluation before stopping."
    eval_pair "$incumbent" "$root/stages/$stage/full_eval/incumbent" 3
    eval_pair "$candidate" "$root/stages/$stage/full_eval/candidate" 3
    printf '%s\n' "$stage $decision" > "$root/STOPPED_AT"
    echo "[STOP] stage=$stage decision=$decision; checkpoints and full reports preserved."
}
run_all() {
    mkdir -p "$root"
    local incumbent="$initial" incumbent_tag="initial"
    eval_pair "$incumbent" "$root/quick/$incumbent_tag" 1
    for i in "${!stage_names[@]}"; do
        local stage="${stage_names[$i]}" probability="${stage_probabilities[$i]}" candidate gate decision
        train_stage "$incumbent" "$stage" "$probability"
        candidate="$(<"$root/stages/$stage/train/checkpoint.txt")"
        eval_pair "$candidate" "$root/quick/$stage" 1
        gate="$root/stages/$stage/gate.json"
        if [[ ! -s "$gate" ]]; then
            "$python" "$repo_dir/scripts/php_student/gate_student_curriculum.py" \
              --incumbent "$root/quick/$incumbent_tag" --candidate "$root/quick/$stage" \
              --output "$gate" --seeds "${quick_seeds[@]}"
        fi
        decision="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$gate")"
        if [[ "$decision" != PASS ]]; then
            full_eval_and_stop "$incumbent" "$candidate" "$stage" "$decision"
            return 0
        fi
        printf '%s\n' "$candidate" > "$root/stages/$stage/PROMOTED_CHECKPOINT"
        incumbent="$candidate"; incumbent_tag="$stage"
    done
    eval_pair "$incumbent" "$root/final_full_eval" 3
    printf '%s\n' "$incumbent" > "$root/FINAL_CHECKPOINT"
    echo "[PASS] All curriculum stages passed; final full evaluation completed."
}
case "$mode" in
  verify)
    check
    "$python" "$repo_dir/scripts/php_student/train_student_history_ab.py" \
      --checkpoint "$initial" --arm single --rollin-probability 0.5 --verify-only ;;
  run) check; run_all ;;
  *) echo "Usage: bash $0 verify|run"; [[ "$mode" == help ]] || exit 2 ;;
esac
