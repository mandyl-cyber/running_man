#!/usr/bin/env python3
"""Compare current-frame and short temporal inputs on Student-induced states.

Offline diagnostic only: Teacher actions are targets, Student closed-loop
states are inputs, and entire motion clips are held out in cross-validation.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.config_types.algo import StudentTeacherModuleConfig
from wbt_training.utils.student_teacher import RoutingColumnDepthStudentTeacherCritic
from probe_observation_vs_state import motion_lengths


def load_policy(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["experiment_config"]["algo"]["config"]
    policy = setup_student_teacher_module(
        num_actor_obs=108, num_teacher_obs=450, num_critic_obs=575,
        num_actions=29, module_config=StudentTeacherModuleConfig(**cfg["module"]),
        device=device, init_noise_std=cfg["init_noise_std"], num_teachers=int(ckpt["num_teachers"]),
        module_cls=functools.partial(RoutingColumnDepthStudentTeacherCritic, routing_column=449),
    )
    policy.load_training_state_dict(ckpt["model_state_dict"], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def collect(trace_dir: Path, policy, device: torch.device, lo: float, hi: float, history: int):
    actors=[]; latents=[]; labels=[]; clips=[]; frames=[]
    lengths=motion_lengths(trace_dir)
    with torch.inference_mode():
      for path in sorted(trace_dir.glob("motion_*.npz")):
        clip=int(path.stem.rsplit("_",1)[1])
        with np.load(path) as d:
          f=d["frame"].astype(np.int64); valid=d["teacher_valid"].astype(bool)
          if len(f)<history: continue
          if not np.all(np.diff(f)==1):
            raise RuntimeError(f"Trace is not contiguous for {path}; cannot form temporal windows")
          depth_map={int(frame):i for i,frame in enumerate(d["depth_frame"])}
          row_map={int(frame):i for i,frame in enumerate(f)}
          duration=max(lengths[clip]-1,1)
          candidates=[i for i,frame in enumerate(f)
                      if lo <= frame/duration < hi and valid[i]
                      and all(int(frame-j) in row_map and int(frame-j) in depth_map for j in range(history))]
          if not candidates: continue
          actor_hist=[]; depth_hist=[]; targets=[]
          for i in candidates:
            frame=int(f[i])
            actor_hist.append(np.stack([d["actor_obs"][row_map[frame-j]] for j in reversed(range(history))]))
            depth_hist.append(np.stack([d["depth"][depth_map[frame-j]] for j in reversed(range(history))]))
            targets.append(d["teacher_action"][i])
          actor=torch.from_numpy(np.asarray(actor_hist,dtype=np.float32)).to(device)
          depth=torch.from_numpy(np.asarray(depth_hist,dtype=np.float32)).flatten(0,1).to(device)
          latent=policy.depth_backbone(depth).reshape(len(candidates),history,-1)
          actors.append(actor.cpu().numpy()); latents.append(latent.cpu().numpy())
          labels.append(np.asarray(targets,dtype=np.float32))
          clips.extend([clip]*len(candidates)); frames.extend([int(f[i]) for i in candidates])
    if not labels: raise RuntimeError("No valid, aligned Student-state temporal samples")
    return (np.concatenate(actors),np.concatenate(latents),np.concatenate(labels),
            np.asarray(clips),np.asarray(frames))


class Regressor(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,output_dim))
    def forward(self,x): return self.net(x)


def fit_cv(actor,latent,target,clips,history,folds,steps,seed,device):
    x=np.concatenate((actor[:,-1],latent[:,-1]),axis=1) if history==1 else np.concatenate((actor,latent),axis=2).reshape(len(actor),-1)
    tensor=torch.from_numpy(x).to(device); y=torch.from_numpy(target).to(device)
    unique=np.unique(clips); rng=np.random.default_rng(seed); rng.shuffle(unique)
    fold_map={int(c):i%folds for i,c in enumerate(unique)}
    rows=[]
    for fold in range(folds):
        test_np=np.asarray([fold_map[int(c)]==fold for c in clips]); train_np=~test_np
        train=torch.from_numpy(np.flatnonzero(train_np)).to(device); test=torch.from_numpy(np.flatnonzero(test_np)).to(device)
        mean=tensor[train].mean(0); std=tensor[train].std(0).clamp_min(1e-3)
        norm=((tensor-mean)/std).clamp(-10,10)
        torch.manual_seed(seed+fold)
        model=Regressor(norm.shape[1],y.shape[1]).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        model.train()
        for _ in range(steps):
            ix=train[torch.randint(len(train),(128,),device=device)]
            loss=(model(norm[ix])-y[ix]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad():
            train_l2=torch.linalg.vector_norm(model(norm[train])-y[train],dim=-1).mean().item()
            test_l2=torch.linalg.vector_norm(model(norm[test])-y[test],dim=-1).mean().item()
        rows.append({"fold":fold,"heldout_clips":[int(c) for c in unique if fold_map[int(c)]==fold],
                     "train_samples":len(train),"heldout_samples":len(test),
                     "train_l2":train_l2,"heldout_l2":test_l2})
        print(f"[history={history} fold={fold}] train={train_l2:.3f} heldout={test_l2:.3f}",flush=True)
        del model,opt
    return {"history":history,"input_dim":int(x.shape[1]),"folds":rows,
            "mean_train_l2":float(np.mean([r["train_l2"] for r in rows])),
            "mean_heldout_l2":float(np.mean([r["heldout_l2"] for r in rows]))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--trace-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--folds",type=int,default=5); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--phase-start",type=float,default=0.25); p.add_argument("--phase-end",type=float,default=0.65)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.output.exists(): p.error(f"Refusing to overwrite {a.output}")
    if not a.checkpoint.is_file() or not a.trace_dir.is_dir() or a.steps<1 or a.folds<2: p.error("Invalid inputs")
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(4); policy=load_policy(a.checkpoint,device)
    actor,latent,target,clips,frames=collect(a.trace_dir,policy,device,a.phase_start,a.phase_end,history=5)
    print(f"[DATA] samples={len(target)} clips={len(np.unique(clips))} history=5 phase={a.phase_start}-{a.phase_end}",flush=True)
    baseline=fit_cv(actor[:,-1:],latent[:,-1:],target,clips,1,a.folds,a.steps,a.seed,device)
    temporal=fit_cv(actor,latent,target,clips,5,a.folds,a.steps,a.seed,device)
    improvement=baseline["mean_heldout_l2"]-temporal["mean_heldout_l2"]
    payload={"contract":"offline_student_closed_loop_teacher_labels_grouped_motion_cv",
      "warning":"Not a closed-loop success estimate; frozen depth features and finite-capacity regressor.",
      "checkpoint":str(a.checkpoint.resolve()),"trace_dir":str(a.trace_dir.resolve()),
      "samples":len(target),"clips":int(len(np.unique(clips))),"phase_window":[a.phase_start,a.phase_end],
      "history_frames":5,"steps_per_fold":a.steps,"seed":a.seed,"baseline_current_frame":baseline,
      "temporal_5frame":temporal,"heldout_l2_improvement":improvement,
      "gate":"PASS" if improvement>=0.05 and all(t["heldout_l2"]<b["heldout_l2"] for b,t in zip(baseline["folds"],temporal["folds"])) else "STOP"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(f"[GATE] {payload['gate']} current={baseline['mean_heldout_l2']:.3f} temporal={temporal['mean_heldout_l2']:.3f} improvement={improvement:.3f}",flush=True)
    print(f"[PASS] {a.output}",flush=True)


if __name__=="__main__": main()
