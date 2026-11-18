"""Matched grouped-CV comparison of zero, frozen, and trainable depth features.

Uses only already-collected states with valid Teacher labels. Offline action prediction
is a mechanism probe; it does not estimate closed-loop Student success.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.config_types.algo import StudentTeacherModuleConfig
from wbt_training.utils.student_teacher import RoutingColumnDepthStudentTeacherCritic

from probe_observation_vs_state import load_samples


def load_backbone(checkpoint_path: Path, samples: dict, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["experiment_config"]["algo"]["config"]
    policy = setup_student_teacher_module(
        num_actor_obs=samples["actor"].shape[1], num_teacher_obs=samples["privileged"].shape[1],
        num_critic_obs=575, num_actions=samples["action"].shape[1],
        module_config=StudentTeacherModuleConfig(**cfg["module"]), device="cpu",
        init_noise_std=cfg["init_noise_std"], num_teachers=int(checkpoint["num_teachers"]),
        module_cls=functools.partial(RoutingColumnDepthStudentTeacherCritic, routing_column=449),
    )
    policy.load_training_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return copy.deepcopy(policy.depth_backbone).to(device)


class ActionRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, actor_dim: int, latent_dim: int,
                 action_dim: int, mode: str, actor_mean: torch.Tensor,
                 actor_std: torch.Tensor, latent_mean: torch.Tensor,
                 latent_std: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.register_buffer("actor_mean", actor_mean)
        self.register_buffer("actor_std", actor_std)
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        self.head = nn.Sequential(nn.Linear(actor_dim + latent_dim, 128), nn.ELU(),
                                  nn.Linear(128, 128), nn.ELU(), nn.Linear(128, action_dim))

    def forward(self, actor: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        actor = ((actor - self.actor_mean) / self.actor_std).clamp(-10, 10)
        if self.mode == "zero":
            latent = actor.new_zeros(actor.shape[0], self.latent_mean.numel())
        elif self.mode == "frozen":
            with torch.no_grad():
                latent = self.backbone(depth)
            latent = ((latent - self.latent_mean) / self.latent_std).clamp(-10, 10)
        else:
            latent = ((self.backbone(depth) - self.latent_mean) / self.latent_std).clamp(-10, 10)
        return self.head(torch.cat((actor, latent), dim=1))


def evaluate(model: ActionRegressor, actor: torch.Tensor, depth: torch.Tensor,
             target: torch.Tensor, ids: torch.Tensor) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in ids.split(128):
            errors.append(torch.linalg.vector_norm(model(actor[batch], depth[batch]) - target[batch], dim=1))
    return float(torch.cat(errors).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--teacher-traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    args = p.parse_args()
    if args.output.exists() or args.steps < 1 or args.folds < 2 or args.backbone_lr <= 0:
        p.error("Use a fresh output and positive training parameters")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU access or --device cpu")
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_samples(args.teacher_traces, 0.25, 0.65)
    clips = np.asarray(sorted(set(map(int, samples["clip"]))))
    if len(clips) < args.folds * 2:
        raise RuntimeError("Need at least two clips per held-out fold")
    # Shuffle at the clip level; every frame from a clip stays in one fold.
    rng = np.random.default_rng(args.seed)
    rng.shuffle(clips)
    fold_by_clip = {int(clip): i % args.folds for i, clip in enumerate(clips)}
    sample_fold = torch.tensor([fold_by_clip[int(clip)] for clip in samples["clip"]], device=device)
    actor = torch.from_numpy(samples["actor"]).to(device)
    depth = torch.from_numpy(samples["depth"]).to(device)
    target = torch.from_numpy(samples["action"]).to(device)
    original_backbone = load_backbone(args.checkpoint, samples, device).eval()
    with torch.no_grad():
        latent = torch.cat([original_backbone(batch) for batch in depth.split(128)])
    fold_rows = []
    for fold in range(args.folds):
        train_ids = torch.where(sample_fold != fold)[0]
        test_ids = torch.where(sample_fold == fold)[0]
        actor_mean = actor[train_ids].mean(0)
        actor_std = actor[train_ids].std(0).clamp_min(1e-3)
        latent_mean = latent[train_ids].mean(0)
        latent_std = latent[train_ids].std(0).clamp_min(1e-3)
        row = {"fold": fold, "heldout_clips": sorted(int(c) for c in clips if fold_by_clip[int(c)] == fold),
               "train_samples": len(train_ids), "heldout_samples": len(test_ids)}
        for mode in ("zero", "frozen", "trainable"):
            torch.manual_seed(args.seed + fold)
            backbone = copy.deepcopy(original_backbone)
            model = ActionRegressor(backbone, actor.shape[1], latent.shape[1], target.shape[1],
                                    mode, actor_mean, actor_std, latent_mean, latent_std).to(device)
            params = [{"params": model.head.parameters(), "lr": 3e-4}]
            if mode == "trainable":
                params.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
            optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
            model.train()
            for _ in range(args.steps):
                selected = train_ids[torch.randint(len(train_ids), (64,), device=device)]
                loss = (model(actor[selected], depth[selected]) - target[selected]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            row[mode] = {"train_l2": evaluate(model, actor, depth, target, train_ids),
                         "heldout_l2": evaluate(model, actor, depth, target, test_ids)}
            del model, optimizer
        fold_rows.append(row)
        print(f"[FOLD {fold}] " + " ".join(
            f"{mode}={row[mode]['heldout_l2']:.3f}" for mode in ("zero", "frozen", "trainable")), flush=True)
    means = {mode: float(np.mean([row[mode]["heldout_l2"] for row in fold_rows]))
             for mode in ("zero", "frozen", "trainable")}
    result = {"contract": "offline_teacher_driven_same_labels_grouped_motion_cv",
              "warning": "Offline regression on Teacher-controlled states is not a closed-loop success estimate.",
              "checkpoint": str(args.checkpoint.resolve()), "traces": str(args.teacher_traces.resolve()),
              "samples": len(target), "clips": len(clips), "folds": args.folds,
              "seed": args.seed, "steps_per_arm_per_fold": args.steps,
              "backbone_lr": args.backbone_lr, "fold_rows": fold_rows, "means": means}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("[MEANS] " + json.dumps(means), flush=True)
    print(f"[PASS] {args.output}", flush=True)


if __name__ == "__main__":
    main()
