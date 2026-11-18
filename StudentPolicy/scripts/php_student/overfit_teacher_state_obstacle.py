"""Small-data diagnostic: can the saved Student fit obstacle-phase Teacher-controlled states?"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.config_types.algo import StudentTeacherModuleConfig

from wbt_training.utils.student_teacher import RoutingColumnDepthStudentTeacherCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1 or args.learning_rate <= 0:
        parser.error("Use a fresh output, positive steps and positive learning rate")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["experiment_config"]["algo"]["config"]
    samples: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in sorted(args.teacher_traces.glob("motion_*.npz")):
        motion_id = int(path.stem.split("_")[-1])
        with np.load(path) as data:
            frame_to_row = {int(frame): i for i, frame in enumerate(data["frame"])}
            for image_index, frame in enumerate(data["depth_frame"]):
                frame = int(frame)
                if frame < 180 or frame >= 320 or frame not in frame_to_row:
                    continue
                row = frame_to_row[frame]
                if not bool(data["teacher_valid"][row]):
                    continue
                samples.append((motion_id, frame, data["actor_obs"][row].copy(),
                                data["depth"][image_index].copy(), data["teacher_action"][row].copy()))
    if len(samples) < 80:
        raise RuntimeError(f"Not enough aligned valid obstacle samples: {len(samples)}")
    actor = torch.from_numpy(np.stack([s[2] for s in samples])).float()
    depth = torch.from_numpy(np.stack([s[3] for s in samples])).float()
    target = torch.from_numpy(np.stack([s[4] for s in samples])).float()
    motion_ids = np.asarray([s[0] for s in samples])
    frame = np.asarray([s[1] for s in samples])
    held_motion = int(motion_ids.max())
    held_motion_mask = motion_ids == held_motion
    validation_mask = (~held_motion_mask) & (frame % 25 == 0)
    training_mask = (~held_motion_mask) & ~validation_mask
    train_idx = torch.from_numpy(np.flatnonzero(training_mask))
    policy = setup_student_teacher_module(
        num_actor_obs=actor.shape[1], num_teacher_obs=450, num_critic_obs=575,
        num_actions=target.shape[1], module_config=StudentTeacherModuleConfig(**cfg["module"]),
        device="cpu", init_noise_std=cfg["init_noise_std"], num_teachers=int(checkpoint["num_teachers"]),
        module_cls=functools.partial(RoutingColumnDepthStudentTeacherCritic, routing_column=449),
    )
    policy.load_training_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    optimizer = torch.optim.AdamW(
        [*policy.student.parameters(), *policy.depth_backbone.parameters()],
        lr=args.learning_rate, weight_decay=0.0,
    )

    @torch.no_grad()
    def evaluate() -> dict:
        prediction = policy.act_inference({"actor_obs": actor, "depth_obs": depth})
        l2 = torch.linalg.vector_norm(prediction - target, dim=-1).numpy()
        mse = F.mse_loss(prediction, target, reduction="none").mean(dim=-1).numpy()
        return {
            key: {"count": int(mask.sum()), "action_l2_mean": float(l2[mask].mean()),
                  "mse_mean": float(mse[mask].mean())}
            for key, mask in (("train", training_mask), ("interleaved_holdout", validation_mask),
                              ("unseen_motion", held_motion_mask))
        }

    before = evaluate()
    for step in range(args.steps):
        selected = train_idx[torch.randint(len(train_idx), (min(32, len(train_idx)),))]
        prediction = policy.act_inference({"actor_obs": actor[selected], "depth_obs": depth[selected]})
        loss = F.mse_loss(prediction, target[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"[OVERFIT] step={step + 1}/{args.steps} batch_mse={float(loss):.5f}", flush=True)
    after = evaluate()
    result = {
        "contract": "offline_teacher_controlled_obstacle_states_pure_mse_diagnostic_not_deployment_policy",
        "checkpoint": str(args.checkpoint),
        "obstacle_frame_window": [180, 320],
        "held_out_motion_id": held_motion,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "before": before,
        "after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
