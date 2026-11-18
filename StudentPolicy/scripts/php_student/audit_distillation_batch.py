"""Recompute Student/Teacher outputs, masked imitation loss and gradients on saved rollout states."""

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

from wbt_training.utils.student_teacher import PhpDistillationPPO, RoutingColumnDepthStudentTeacherCritic


def load_rows(trace_dir: Path, frame_lo: int, frame_hi: int, per_motion: int) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {key: [] for key in
        ("actor_obs", "teacher_obs", "depth", "student_action", "teacher_action", "frame")}
    for path in sorted(trace_dir.glob("motion_*.npz")):
        with np.load(path) as data:
            frame_to_row = {int(frame): i for i, frame in enumerate(data["frame"])}
            picked = [i for i, frame in enumerate(data["depth_frame"])
                      if frame_lo <= frame < frame_hi and int(frame) in frame_to_row][:per_motion]
            for i in picked:
                j = frame_to_row[int(data["depth_frame"][i])]
                for key in ("actor_obs", "teacher_obs", "student_action", "teacher_action", "frame"):
                    rows[key].append(np.asarray(data[key][j]).copy())
                rows["depth"].append(np.asarray(data["depth"][i]).copy())
    if not rows["frame"]:
        raise ValueError(f"No depth-aligned frames in {trace_dir} [{frame_lo}, {frame_hi})")
    return {key: np.stack(value) for key, value in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-traces", type=Path, required=True)
    parser.add_argument("--student-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(2)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["experiment_config"]["algo"]["config"]
    teacher_rows = load_rows(args.teacher_traces, 200, 300, 8)
    lost_rows = load_rows(args.student_traces, 300, 500, 8)
    rows = {key: np.concatenate((teacher_rows[key], lost_rows[key])) for key in teacher_rows}
    valid = np.any(rows["teacher_action"] != 0.0, axis=1)
    if not valid[:len(teacher_rows["frame"])].all() or valid[len(teacher_rows["frame"]):].any():
        raise RuntimeError("Expected valid Teacher-controlled rows followed by expert-lost Student rows")
    actor = torch.from_numpy(rows["actor_obs"]).float()
    teacher = torch.from_numpy(rows["teacher_obs"]).float()
    depth = torch.from_numpy(rows["depth"]).float()
    route_col = teacher.shape[1] - 1
    route = teacher[:, route_col].numpy()
    if not np.array_equal(route < 0, ~valid):
        raise RuntimeError("Teacher routing index and zero-action validity mask disagree")
    policy = setup_student_teacher_module(
        num_actor_obs=actor.shape[1], num_teacher_obs=teacher.shape[1], num_critic_obs=575,
        num_actions=rows["teacher_action"].shape[1],
        module_config=StudentTeacherModuleConfig(**cfg["module"]), device="cpu",
        init_noise_std=cfg["init_noise_std"], num_teachers=int(ckpt["num_teachers"]),
        module_cls=functools.partial(RoutingColumnDepthStudentTeacherCritic, routing_column=route_col),
    )
    policy.load_training_state_dict(ckpt["model_state_dict"], strict=True)
    policy.set_loaded_teachers([True] * int(ckpt["num_teachers"]))
    policy.eval()
    with torch.no_grad():
        teacher_pred = policy.teacher_act(teacher)
    student_pred = policy.act_inference({"actor_obs": actor, "depth_obs": depth})
    # Exactly the PHP subclass's per-sample loss, using its production method.
    algo = object.__new__(PhpDistillationPPO)
    algo.distill_loss_fn = F.mse_loss
    per_sample = algo._dagger_loss_per_sample(student_pred, teacher_pred)
    raw = F.mse_loss(student_pred, teacher_pred, reduction="none").mean(dim=-1)
    manual = raw * torch.from_numpy(valid).float()
    max_loss_error = float((per_sample - manual).abs().max())
    if max_loss_error > 1e-7:
        raise RuntimeError(f"Production and manual loss mismatch: {max_loss_error}")
    per_sample.mean().backward()
    student_grad = sum(float(p.grad.square().sum()) for p in policy.student.parameters() if p.grad is not None) ** 0.5
    depth_grad = sum(float(p.grad.square().sum()) for p in policy.depth_backbone.parameters() if p.grad is not None) ** 0.5
    teacher_grad = sum(float(p.grad.square().sum()) for p in policy.teachers.parameters() if p.grad is not None) ** 0.5
    if not (student_grad > 0 and depth_grad > 0 and np.isfinite(student_grad) and np.isfinite(depth_grad)):
        raise RuntimeError("Imitation loss did not yield finite nonzero Student/depth gradients")
    result = {
        "contract": "actual_checkpoint_and_saved_depth_aligned_rollout_states_no_optimizer_step",
        "checkpoint": str(args.checkpoint),
        "valid_teacher_samples": int(valid.sum()),
        "expert_lost_samples": int((~valid).sum()),
        "teacher_action_recompute_max_abs": float(np.abs(teacher_pred.numpy() - rows["teacher_action"]).max()),
        "student_action_recompute_max_abs": float(np.abs(student_pred.detach().numpy() - rows["student_action"]).max()),
        "loss_manual_max_abs": max_loss_error,
        "valid_loss_mean": float(per_sample.detach().numpy()[valid].mean()),
        "lost_loss_max_abs": float(per_sample.detach().numpy()[~valid].max()),
        "student_grad_l2": student_grad,
        "depth_backbone_grad_l2": depth_grad,
        "teacher_grad_l2": teacher_grad,
        "ppo_coef_checkpoint": float(ckpt["ppo_coef"]),
        "dagger_loss_coef": float(cfg["dagger_loss_coef"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
