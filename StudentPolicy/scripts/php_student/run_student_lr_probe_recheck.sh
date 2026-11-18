#!/usr/bin/env bash
# Replicate the 1-episode LR-probe comparisons with 3 episodes per motion.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PHP_PYTHON:-/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python}"
baseline="/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion"
candidate="$(<"$repo_dir/reports/php_student/lr_probe_kl003_v1/train_dir.txt")"
root="$repo_dir/reports/php_student/lr_probe_kl003_recheck_v1"

for group in baseline candidate; do
  if [[ "$group" == baseline ]]; then train_dir="$baseline"; else train_dir="$candidate"; fi
  for id in 00500 01000 02500; do
    out="$root/$group/model_$id"
    if [[ -s "$out/full_clip_seed43.json" ]]; then echo "[SKIP] $out"; continue; fi
    STUDENT_CHECKPOINT="$train_dir/model_${id}.pt" \
    STUDENT_EVAL_REPORT_DIR="$out" STUDENT_EVAL_EPISODES_PER_MOTION=3 \
    STUDENT_EVAL_NUM_ENVS=40 STUDENT_EVAL_MAX_STEPS=12000 \
    STUDENT_EVAL_SEED=43 STUDENT_EVAL_OBSTACLE_SEED=43 \
    STUDENT_EVAL_DIAGNOSTICS=1 STUDENT_EVAL_TAKEOVER_MODE=none PHP_EVAL_PROFILE=clean \
      bash "$repo_dir/scripts/php_student/eval_student_full.sh"
  done
done

"$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for i in (500,1000,2500):
    row={"iteration":i}
    for group in ("baseline","candidate"):
        p=root/group/f"model_{i:05d}"/"full_clip_seed43.json"
        d=json.loads(p.read_text()); assert d["complete"]
        clips=d["motions"]; n=sum(x["episodes"] for x in clips); s=sum(x["successes"] for x in clips)
        vals=[x.get("diagnostic",{}).get("failure_progress_mean") for x in clips if x["successes"]<x["episodes"]]
        vals=[v for v in vals if v is not None]
        row[group]={"successes":s,"episodes":n,"rate":s/n,"failure_progress":sum(vals)/len(vals) if vals else 1.0}
    rows.append(row)
out={"episodes_per_motion":3,"seed":43,"rows":rows}
(root/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows: print(r)
print(f"[PASS] {root/'summary.json'}")
PY
