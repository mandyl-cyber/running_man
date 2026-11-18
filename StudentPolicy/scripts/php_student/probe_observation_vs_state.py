"""Offline, same-state/same-label observability probe on recorded rollouts.

This is a diagnostic, not a deployable Student or an online success estimate.
Each fold holds out an entire motion clip to avoid adjacent-frame leakage.
Only frames with valid Teacher labels are included; the caller must record
whether rollouts were Student-controlled or Teacher-controlled.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.config_types.algo import StudentTeacherModuleConfig
from wbt_training.utils.student_teacher import RoutingColumnDepthStudentTeacherCritic


def motion_lengths(trace_dir: Path) -> dict[int, int]:
    names_path = trace_dir.parent / "motion_names.txt"
    bundle = Path(os.environ.get("PHP_BUNDLE", "/home/sjtu/.cache/php-parkour/motions-low-climb-76"))
    if not names_path.is_file() or not bundle.is_dir():
        raise FileNotFoundError(f"Need {names_path} and motion bundle {bundle} for true reference phase")
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    lengths = {}
    for clip, name in enumerate(names):
        with np.load(bundle / f"{name}_motion.npz") as motion:
            lengths[clip] = len(motion["joint_pos"])
    return lengths


def load_samples(trace_dir: Path, lo: float, hi: float) -> dict[str, np.ndarray]:
    items: list[tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    lengths = motion_lengths(trace_dir)
    paths=sorted(trace_dir.glob("motion_*.npz"))
    for file_idx, path in enumerate(paths,1):
        print(f"[TRACE LOAD] {file_idx}/{len(paths)} {path.name}",flush=True)
        clip = int(path.stem.split("_")[-1])
        with np.load(path) as data:
            frames = data["frame"]
            frame_to_row = {int(frame): i for i, frame in enumerate(frames)}
            duration = max(lengths[clip] - 1, 1)
            for image_idx, frame in enumerate(data["depth_frame"]):
                frame = int(frame)
                row = frame_to_row.get(frame)
                if row is None or not lo <= frame / duration < hi or not data["teacher_valid"][row]:
                    continue
                items.append((clip, frame, frame / duration, data["actor_obs"][row].copy(),
                              data["depth"][image_idx].copy(), data["teacher_obs"][row].copy(),
                              data["teacher_action"][row].copy()))
    if len(items) < 100 or len({item[0] for item in items}) < 3:
        raise RuntimeError("Need >=100 valid aligned frames from >=3 Teacher-driven clips")
    return {"clip": np.asarray([x[0] for x in items]), "frame": np.asarray([x[1] for x in items]),
            "phase": np.asarray([x[2] for x in items]),
            "actor": np.stack([x[3] for x in items]).astype(np.float32),
            "depth": np.stack([x[4] for x in items]).astype(np.float32),
            "privileged": np.stack([x[5] for x in items]).astype(np.float32),
            "action": np.stack([x[6] for x in items]).astype(np.float32)}


def frozen_depth_features(checkpoint_path: Path, samples: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["experiment_config"]["algo"]["config"]
    actor = torch.from_numpy(samples["actor"])
    depth = torch.from_numpy(samples["depth"])
    policy = setup_student_teacher_module(
        num_actor_obs=actor.shape[1], num_teacher_obs=samples["privileged"].shape[1],
        num_critic_obs=575, num_actions=samples["action"].shape[1],
        module_config=StudentTeacherModuleConfig(**cfg["module"]), device="cpu",
        init_noise_std=cfg["init_noise_std"], num_teachers=int(checkpoint["num_teachers"]),
        module_cls=functools.partial(RoutingColumnDepthStudentTeacherCritic, routing_column=449),
    )
    policy.load_training_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    with torch.no_grad():
        vision = policy._student_input(actor, depth).detach()
        existing_l2 = torch.linalg.vector_norm(
            policy.act_inference({"actor_obs": actor, "depth_obs": depth})
            - torch.from_numpy(samples["action"]), dim=-1,
        )
    return vision, existing_l2


def fit_fold(features: torch.Tensor, labels: torch.Tensor, train: torch.Tensor,
             test: torch.Tensor, steps: int, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    mean = features[train].mean(0)
    std = features[train].std(0).clamp_min(1e-3)
    x = ((features - mean) / std).clamp(-10, 10)
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ELU(), nn.Linear(128, 128),
                          nn.ELU(), nn.Linear(128, labels.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_ids = torch.where(train)[0]
    for _ in range(steps):
        selected = train_ids[torch.randint(len(train_ids), (min(64, len(train_ids)),))]
        loss = (model(x[selected]) - labels[selected]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        err = torch.linalg.vector_norm(model(x) - labels, dim=-1)
    return {"train_l2": float(err[train].mean()), "heldout_l2": float(err[test].mean()),
            "train_n": int(train.sum()), "heldout_n": int(test.sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--teacher-traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--phase-start", type=float, default=0.25)
    p.add_argument("--phase-end", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists() or args.steps < 1 or not 0 <= args.phase_start < args.phase_end <= 1:
        p.error("Use a fresh output, positive steps, and a valid phase window")
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_samples(args.teacher_traces, args.phase_start, args.phase_end)
    vision, existing_error = frozen_depth_features(args.checkpoint, samples)
    actor = torch.from_numpy(samples["actor"])
    # Ground-truth reference phase is an offline timing oracle, never an input
    # available to the deployed student. Keep the same labels and held-out clips.
    phase = torch.from_numpy(samples["phase"].astype(np.float32))[:, None]
    features = {"actor_only": actor,
                "actor_plus_zero_depth": torch.cat((actor, torch.zeros_like(vision[:, actor.shape[1]:])), dim=1),
                "student_actor_plus_frozen_depth": vision,
                "actor_plus_oracle_phase": torch.cat((actor, phase), dim=1),
                "student_frozen_depth_plus_oracle_phase": torch.cat((vision, phase), dim=1),
                "privileged_teacher_obs": torch.from_numpy(samples["privileged"])}
    labels = torch.from_numpy(samples["action"])
    clips = sorted(set(map(int, samples["clip"])))
    folds = []
    for clip in clips:
        test = torch.from_numpy(samples["clip"] == clip)
        train = ~test
        row = {"heldout_clip_id": clip, "existing_student_l2": float(existing_error[test].mean())}
        for name, values in features.items():
            row[name] = fit_fold(values, labels, train, test, args.steps, args.seed)
        folds.append(row)
        print(f"[FOLD {clip}] " + " ".join(
            f"{name}={row[name]['heldout_l2']:.3f}"
            for name in features), flush=True)
    # A contiguous, within-motion held-out block tests interpolation without
    # claiming that adjacent frames are statistically independent.
    block_start = args.phase_start + 0.5 * (args.phase_end - args.phase_start)
    block_end = block_start + 0.25 * (args.phase_end - args.phase_start)
    block_test = torch.from_numpy((samples["phase"] >= block_start) & (samples["phase"] < block_end))
    block_train = ~block_test
    if int(block_test.sum()) < 20 or int(block_train.sum()) < 80:
        raise RuntimeError("Not enough samples for contiguous held-out phase block")
    temporal_block = {name: fit_fold(values, labels, block_train, block_test, args.steps, args.seed)
                      for name, values in features.items()}
    result = {"contract": "offline_student_closed_loop_same_state_teacher_label_leave_one_clip_out",
              "interpretation_warning": "Teacher observations and oracle phase are privileged diagnostics, not deployable inputs. Frozen depth features cannot distinguish missing image information from a weak saved encoder. Phase alone does not prove visual timing is learnable.",
              "checkpoint": str(args.checkpoint.resolve()),
              "teacher_traces": str(args.teacher_traces.resolve()),
              "phase_window": [args.phase_start, args.phase_end],
              "steps_per_fold": args.steps, "seed": args.seed,
              "samples": len(labels), "clips": clips,
              "folds": folds,
              "within_clip_contiguous_block": {"phase_window": [block_start, block_end],
                                                "results": temporal_block},
              "means": {name: float(np.mean([row[name]["heldout_l2"] for row in folds]))
                        for name in features}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["means"], indent=2), flush=True)
    print("[WITHIN-CLIP BLOCK] " + " ".join(
        f"{name}={metrics['heldout_l2']:.3f}" for name, metrics in temporal_block.items()), flush=True)
    print(f"[PASS] {args.output}", flush=True)


if __name__ == "__main__":
    main()
