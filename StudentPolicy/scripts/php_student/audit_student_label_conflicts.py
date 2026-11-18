#!/usr/bin/env python3
"""Quantify cross-motion Teacher-label conflict among similar Student observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_observation_vs_state import frozen_depth_features, load_samples


def nearest_report(features: np.ndarray, actions: np.ndarray, clips: np.ndarray,
                   name: str, k: int, seed: int) -> dict:
    source_sample_count=len(features)
    # Keep a deterministic, phase-stratified subset to bound the O(N^2 D)
    # audit cost. Every motion contributes equally, even if its trace is longer.
    rng=np.random.default_rng(seed)
    keep=[]
    for clip in np.unique(clips):
        ids=np.flatnonzero(clips == clip)
        if len(ids)>60:
            ids=ids[np.linspace(0,len(ids)-1,60,dtype=int)]
        keep.extend(ids.tolist())
    keep=np.asarray(sorted(keep),dtype=np.int64)
    features=features[keep]; actions=actions[keep]; clips=clips[keep]
    mean=features.mean(axis=0,keepdims=True)
    std=np.maximum(features.std(axis=0,keepdims=True),1e-3)
    x=torch.from_numpy(np.clip((features-mean)/std,-10,10).astype(np.float32))
    # Torch-only exact search. Retrieve a generous candidate set because
    # same-motion neighbors are excluded from the cross-motion comparison.
    x2=(x*x).sum(dim=1)
    d2=(x2[:,None]+x2[None,:]-2.0*(x@x.T)).clamp_min_(0.0)
    d2.fill_diagonal_(float("inf"))
    n_candidates=min(len(x)-1,max(k+256,512))
    candidate_d2,candidate_ids=torch.topk(d2,n_candidates,dim=1,largest=False,sorted=True)
    indices=candidate_ids.numpy(); distances=candidate_d2.sqrt().numpy()
    cross_errors=[]; cross_distances=[]; within_errors=[]
    per_clip_cross={int(c):[] for c in np.unique(clips)}
    for i in range(len(x)):
        other=[j for j in indices[i] if clips[j] != clips[i]][:k]
        same=[j for j in indices[i] if clips[j] == clips[i]][:k]
        if len(other)>=k:
            neighbor_error=np.linalg.norm(actions[other]-actions[i],axis=1)
            cross_errors.append(neighbor_error)
            per_clip_cross[int(clips[i])].append(float(neighbor_error.mean()))
            cross_distances.append(np.linalg.norm(x[other]-x[i],axis=1))
        if len(same)>=k:
            within_errors.append(np.linalg.norm(actions[same]-actions[i],axis=1))
    if not cross_errors:
        raise RuntimeError("No cross-motion neighbors found")
    cross=np.concatenate(cross_errors); dist=np.concatenate(cross_distances)
    within=np.concatenate(within_errors) if within_errors else np.asarray([])
    random_errors=[]
    sample_n=min(100000,len(cross))
    for _ in range(sample_n):
        i=int(rng.integers(0,len(actions)))
        eligible=np.flatnonzero(clips != clips[i])
        j=int(rng.choice(eligible))
        random_errors.append(np.linalg.norm(actions[i]-actions[j]))
    random_errors=np.asarray(random_errors)
    clip_means=np.asarray([np.mean(v) for v in per_clip_cross.values() if v],dtype=np.float64)
    bootstrap=np.mean(rng.choice(clip_means,size=(5000,len(clip_means)),replace=True),axis=1)
    return {
      "features":name,"samples":len(x),"source_samples":int(source_sample_count),"clips":int(len(np.unique(clips))),"k":k,
      "cross_motion_neighbor_distance_median":float(np.median(dist)),
      "cross_motion_action_l2_mean":float(cross.mean()),
      "cross_motion_action_l2_clip_balanced_mean":float(clip_means.mean()),
      "cross_motion_action_l2_clip_bootstrap_95ci":[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],
      "cross_motion_action_l2_median":float(np.median(cross)),
      "cross_motion_action_l2_p90":float(np.quantile(cross,.9)),
      "within_motion_action_l2_mean":float(within.mean()) if len(within) else None,
      "random_pair_action_l2_mean":float(random_errors.mean()),
      "conflict_ratio_to_random":float(cross.mean()/max(random_errors.mean(),1e-8)),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--trace-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase-start",type=float,default=.25)
    p.add_argument("--phase-end",type=float,default=.65)
    p.add_argument("--k",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.output.exists(): p.error(f"Refusing to overwrite {a.output}")
    torch.set_num_threads(4)
    samples=load_samples(a.trace_dir,a.phase_start,a.phase_end)
    print(f"[LOAD] samples={len(samples['action'])} clips={len(np.unique(samples['clip']))}",flush=True)
    vision,_=frozen_depth_features(a.checkpoint,samples)
    print("[FEATURES] frozen Student features ready",flush=True)
    actor=samples["actor"]
    student=np.concatenate((actor,vision[:,actor.shape[1]:].numpy()),axis=1)
    # The routing index is a clip identity cue, not a deployable physical state.
    routing_column=449
    if samples["privileged"].shape[1] <= routing_column:
        raise ValueError("Teacher observation has no routing column 449")
    privileged=np.delete(samples["privileged"],routing_column,axis=1)
    clips=samples["clip"]; actions=samples["action"]
    results=[]
    for name,x in (("actor_only",actor),("actor_plus_frozen_depth",student),("privileged_without_routing_id",privileged)):
        row=nearest_report(x,actions,clips,name,a.k,a.seed); results.append(row)
        print(f"[{name}] cross_action={row['cross_motion_action_l2_mean']:.3f} "
              f"within={row['within_motion_action_l2_mean']} random={row['random_pair_action_l2_mean']:.3f} "
              f"near_dist={row['cross_motion_neighbor_distance_median']:.3f}",flush=True)
    result={"contract":"student_closed_loop_state_cross_motion_nearest_neighbor_label_conflict",
      "warning":"Nearest-neighbor association is diagnostic, not proof of irreducible partial observability.",
      "checkpoint":str(a.checkpoint.resolve()),"trace_dir":str(a.trace_dir.resolve()),
      "phase_window":[a.phase_start,a.phase_end],"phase_denominator":"bundle_joint_pos_frames_minus_one",
      "removed_teacher_routing_column":routing_column,"k":a.k,"samples":len(actions),
      "clip_count":int(len(np.unique(clips))),"results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"[PASS] {a.output}",flush=True)


if __name__=="__main__": main()
