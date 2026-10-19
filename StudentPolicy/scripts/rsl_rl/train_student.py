# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 训练 G1 全身轨迹跟踪策略。

主要流程：
1. 启动 Isaac Sim/Isaac Lab app。
2. 读取命令行参数，配置 motion、manifest、terrain 和 PPO。
3. 创建 Tracking-Flat-G1-v0 环境。
4. 用 RSL-RL OnPolicyRunner 训练并保存 checkpoint。

这个脚本只负责训练；可视化 checkpoint 请用 `scripts/rsl_rl/play.py`。
"""

"""先启动 Isaac Sim 仿真程序。"""

import argparse
import json
import math
import os
import pathlib
import pickle
import sys

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/running_man_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/running_man_cache/matplotlib")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/running_man_cache/warp")

STUDENT_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = STUDENT_ROOT.parent
STUDENT_PACKAGE_ROOT = STUDENT_ROOT / "source" / "student_policy"
TEACHER_ROOT = WORKSPACE_ROOT / "new_teacherpolicy"
TEACHER_PACKAGE_ROOT = TEACHER_ROOT / "source" / "whole_body_tracking"
for package_root in (STUDENT_PACKAGE_ROOT, TEACHER_PACKAGE_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher

# 本地脚本导入
import cli_args  # isort: skip
import dataset_args  # isort: skip

G1_CANONICAL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_CANONICAL_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# 本地训练流程专用的命令行参数。
parser = argparse.ArgumentParser(description="Train a student policy with DAgger + PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-Student-v0", help="Name of the student task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the local motion npz file.")
parser.add_argument("--motion_files", nargs="+", default=None, help="Paths to local motion npz files.")
parser.add_argument("--motion_dir", type=str, default=None, help="Directory containing local motion npz files.")
parser.add_argument(
    "--motion_source_order",
    type=str,
    default="g1_canonical",
    choices=["g1_canonical", "robot"],
    help="Source joint/body order used by the local motion npz file.",
)
parser.add_argument("--manifest_file", type=str, default=None, help="Path to the local batch_manifest.json file.")
parser.add_argument("--terrain_file", type=str, default=None, help="Path to the local terrain USD file.")
parser.add_argument(
    "--terrain_use_manifest_pose",
    action="store_true",
    help="Spawn --terrain_file at manifest terrain_world_pose instead of the origin.",
)
parser.add_argument("--motion_root_body_idx", type=int, default=0, help="Root body index used for manifest alignment.")
parser.add_argument(
    "--fixed_start_frame",
    type=int,
    default=-1,
    help="Motion frame used for every episode reset. Use -1 to keep adaptive random start-frame sampling.",
)
parser.add_argument("--env_spacing", type=float, default=None, help="Environment spacing override.")
parser.add_argument("--command_x_min", type=float, default=1.0, help="Minimum commanded forward velocity.")
parser.add_argument("--command_x_max", type=float, default=1.0, help="Maximum commanded forward velocity.")
parser.add_argument("--command_y_min", type=float, default=0.0, help="Minimum commanded lateral velocity.")
parser.add_argument("--command_y_max", type=float, default=0.0, help="Maximum commanded lateral velocity.")
shortcut_group = parser.add_mutually_exclusive_group()
shortcut_group.add_argument(
    "--multi",
    type=str,
    default=None,
    metavar="DATASET",
    help="Shortcut for multi-trajectory student training under new_teacherpolicy/data/DATASET.",
)
shortcut_group.add_argument(
    "--simple",
    type=str,
    default=None,
    metavar="DATASET",
    help="Shortcut for single-trajectory student training using the first npz under new_teacherpolicy/data/DATASET.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    required=True,
    help="Frozen teacher checkpoint used by Tracking-Flat-G1-Student-v0 DAgger training.",
)
dataset_args.add_dataset_args(parser)

# 追加 RSL-RL 命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加 Isaac Sim AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.multi is not None or args_cli.simple is not None:
    shortcut_name = args_cli.multi if args_cli.multi is not None else args_cli.simple
    if args_cli.dataset_dir is None:
        args_cli.dataset_dir = str(TEACHER_ROOT / "data")
    if args_cli.dataset_name is None:
        args_cli.dataset_name = shortcut_name
    if args_cli.task is None:
        args_cli.task = "Tracking-Flat-G1-Student-v0"
    if args_cli.max_iterations is None:
        args_cli.max_iterations = 30000
    if args_cli.logger is None:
        args_cli.logger = "tensorboard"
    args_cli.headless = True
    if args_cli.simple is not None and args_cli.dataset_motion_index is None:
        args_cli.dataset_motion_index = 0

dataset_args.apply_dataset_defaults(args_cli, mode="train")

# 录制视频时必须启用 camera
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv，避免 Hydra 解析到非 Hydra 参数
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse / Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim 启动后，再导入依赖仿真的模块。"""

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from datetime import datetime

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入扩展以注册环境任务
import whole_body_tracking.tasks  # noqa: F401
import student_policy.tasks  # noqa: F401
from student_policy.dagger_runner import StudentDaggerOnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data: object):
    """把 Python 配置对象保存到日志目录，方便精确复现实验。"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """把角度制 yaw 转成 Isaac/Usd 使用的 WXYZ 四元数。"""
    yaw_rad = math.radians(float(yaw_deg))
    half_yaw = 0.5 * yaw_rad
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _set_sensor_terrain_target(env_cfg: ManagerBasedRLEnvCfg, sensor_name: str):
    """Point ray-based perception sensors at the per-env terrain USD.

    Some Isaac Lab ray-caster variants only support a single mesh prim. When a
    local terrain is loaded, replace the default ground target instead of adding
    a second mesh target.
    """
    sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
    if sensor_cfg is None:
        return

    terrain_target = "{ENV_REGEX_NS}/Terrain"
    if hasattr(sensor_cfg.__class__, "RaycastTargetCfg"):
        sensor_cfg.mesh_prim_paths = [
            sensor_cfg.__class__.RaycastTargetCfg(prim_expr=terrain_target, track_mesh_transforms=False)
        ]
    else:
        sensor_cfg.mesh_prim_paths = [terrain_target]
    print(f"[INFO]: Using {sensor_name} terrain target: {terrain_target}")


def _add_height_scan_terrain_target(env_cfg: ManagerBasedRLEnvCfg):
    _set_sensor_terrain_target(env_cfg, "height_scanner")
    _set_sensor_terrain_target(env_cfg, "depth_camera")


def _load_manifest_entry(manifest_file: str | None, motion_file: str) -> dict | None:
    """在 manifest 中找到某个 motion npz 对应的 trajectory 条目。"""
    if manifest_file is None:
        return None

    manifest_path = os.path.abspath(manifest_file)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Invalid manifest file path: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        payload = json.load(f)

    trajectories = payload.get("trajectories", [])
    if not trajectories:
        raise ValueError(f"No trajectories found in manifest: {manifest_path}")

    motion_name = os.path.basename(motion_file)
    motion_stem = os.path.splitext(motion_name)[0]
    for item in trajectories:
        trajectory_path = item.get("trajectory_path", "")
        trajectory_name = item.get("trajectory_name", "")
        if os.path.basename(trajectory_path) == motion_name:
            return item
        if trajectory_name and motion_stem == trajectory_name:
            return item

    if len(trajectories) == 1:
        print(f"[WARN] No exact manifest match for {motion_name}; using the only trajectory entry.")
        return trajectories[0]

    raise ValueError(f"Could not find manifest trajectory entry matching motion file: {motion_name}")


def _get_skill_output_start_frame(manifest_entry: dict) -> int:
    """返回生成技能轨迹开始执行的第一帧。"""
    for segment in manifest_entry.get("segments", []):
        if segment.get("mode") == "skill_execution":
            return int(segment["output_start_frame"])
    raise ValueError("Manifest entry has no skill_execution segment.")


def _rotate_xyz_by_yaw_deg(xyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    """把 3D 平移向量绕世界 z 轴旋转 yaw 角度。"""
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.array(
        [
            cos_yaw * float(xyz[0]) - sin_yaw * float(xyz[1]),
            sin_yaw * float(xyz[0]) + cos_yaw * float(xyz[1]),
            float(xyz[2]),
        ],
        dtype=np.float32,
    )


def _compute_manifest_motion_offset(
    manifest_entry: dict, motion_file: str, root_body_idx: int, use_manifest_terrain_pose: bool
) -> tuple[float, float, float]:
    """把 npz 根节点位置对齐到 manifest 里的 skill anchor。

    动作生成器和 Isaac 场景可能使用不同世界原点。这里计算出的 offset 会平移
    参考轨迹里的所有 body 位置，让 skill 第一帧从选定 terrain 上的 manifest anchor 开始。
    """
    data = np.load(motion_file)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    skill_output_start = _get_skill_output_start_frame(manifest_entry)
    if skill_output_start >= len(body_pos_w):
        raise ValueError(
            f"skill_execution.output_start_frame={skill_output_start} exceeds motion length {len(body_pos_w)}."
        )

    skill_anchor = manifest_entry["skill_anchor"]
    target_root = np.asarray(skill_anchor["root_translation"], dtype=np.float32)
    target_root_frame = "manifest skill world"

    if (
        use_manifest_terrain_pose
        and not bool(skill_anchor.get("fixed_world", False))
        and "terrain_world_pose" in manifest_entry
    ):
        terrain_pose = manifest_entry["terrain_world_pose"]
        terrain_translation = np.asarray(terrain_pose.get("translation", (0.0, 0.0, 0.0)), dtype=np.float32)
        terrain_yaw_deg = float(terrain_pose.get("yaw_deg", 0.0))
        target_root = terrain_translation + _rotate_xyz_by_yaw_deg(target_root, terrain_yaw_deg)
        target_root_frame = "manifest terrain world"

    observed_root = body_pos_w[skill_output_start, root_body_idx]
    offset = target_root - observed_root
    print(f"[INFO]: Target motion root ({target_root_frame}): {target_root.tolist()}")
    print(f"[INFO]: Applying manifest motion offset: {offset.tolist()}")
    return tuple(float(x) for x in offset)


def _add_manifest_terrain(
    env_cfg: ManagerBasedRLEnvCfg, terrain_file: str, manifest_entry: dict | None, use_manifest_pose: bool
):
    """给每个并行环境挂载一个本地 terrain USD。"""
    terrain_path = os.path.abspath(terrain_file)
    if not os.path.isfile(terrain_path):
        raise FileNotFoundError(f"Invalid terrain file path: {terrain_path}")

    terrain_pose = manifest_entry.get("terrain_world_pose", {}) if use_manifest_pose and manifest_entry is not None else {}
    translation = terrain_pose.get("translation", (0.0, 0.0, 0.0))
    yaw_deg = terrain_pose.get("yaw_deg", 0.0)
    terrain_pos = tuple(float(x) for x in translation)
    terrain_rot = _yaw_quat_wxyz(float(yaw_deg))

    env_cfg.scene.terrain_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain",
        spawn=sim_utils.UsdFileCfg(
            usd_path=terrain_path,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=terrain_pos, rot=terrain_rot),
        collision_group=0,
    )
    _add_height_scan_terrain_target(env_cfg)
    print(f"[INFO]: Using local terrain USD: {terrain_path}")
    if use_manifest_pose:
        print(f"[INFO]: Terrain pose from manifest: pos={terrain_pos}, yaw_deg={float(yaw_deg)}")
    else:
        print(f"[INFO]: Terrain pose: pos={terrain_pos}, yaw_deg={float(yaw_deg)}")


def _use_fixed_motion_start(env_cfg: ManagerBasedRLEnvCfg, start_frame: int):
    """强制每次 reset 都从同一个参考帧开始。

    这适合确定性 debug/play。正常训练应保持 `--fixed_start_frame -1`，
    让策略看到更多起始状态。
    """
    env_cfg.commands.motion.fixed_start_frame = start_frame
    env_cfg.commands.motion.pose_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.commands.motion.velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    print(f"[INFO]: Fixed motion reset frame: {start_frame}")


def _resolve_motion_files() -> list[str]:
    """从 --motion_file、--motion_files、--motion_dir 收集动作文件。

    返回值会转为绝对路径，检查文件存在，目录输入会排序，并去重。
    """
    motion_files = []
    if args_cli.motion_file is not None:
        motion_files.append(args_cli.motion_file)
    if args_cli.motion_files is not None:
        motion_files.extend(args_cli.motion_files)
    if args_cli.motion_dir is not None:
        motion_dir = os.path.abspath(args_cli.motion_dir)
        if not os.path.isdir(motion_dir):
            raise FileNotFoundError(f"Invalid motion directory path: {motion_dir}")
        motion_files.extend(
            os.path.join(motion_dir, name)
            for name in sorted(os.listdir(motion_dir))
            if name.endswith(".npz")
        )

    if not motion_files:
        raise ValueError("Provide --motion_file, --motion_files, or --motion_dir for local training.")

    resolved = []
    seen = set()
    for motion_file in motion_files:
        motion_path = os.path.abspath(motion_file)
        if motion_path in seen:
            continue
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Invalid motion file path: {motion_path}")
        resolved.append(motion_path)
        seen.add(motion_path)
    return resolved


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL agent 开始训练。"""
    # 用非 Hydra 命令行参数覆盖默认配置。
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing if args_cli.env_spacing is not None else env_cfg.scene.env_spacing
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.teacher_checkpoint is not None:
        agent_cfg.teacher_checkpoint = args_cli.teacher_checkpoint
    env_cfg.observations.policy.command.params = {
        "x_range": (args_cli.command_x_min, args_cli.command_x_max),
        "y_range": (args_cli.command_y_min, args_cli.command_y_max),
    }

    # 设置环境随机种子。部分随机化会在环境初始化时发生，所以这里先设置。
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # 加载一个或多个本地参考动作；多轨迹采样在 MotionCommand 内部处理。
    motion_files = _resolve_motion_files()
    print("[INFO]: Using local motion files:")
    for motion_file in motion_files:
        print(f"  - {motion_file}")
    env_cfg.commands.motion.motion_file = motion_files[0]
    env_cfg.commands.motion.motion_files = motion_files
    if args_cli.motion_source_order == "g1_canonical":
        env_cfg.commands.motion.source_joint_names = G1_CANONICAL_JOINT_NAMES
        env_cfg.commands.motion.source_body_names = G1_CANONICAL_BODY_NAMES

    # 一个 manifest 可以包含多条 trajectory；每个 npz 按文件名或 trajectory_name 匹配。
    manifest_entries = [_load_manifest_entry(args_cli.manifest_file, motion_file) for motion_file in motion_files]
    if any(entry is not None for entry in manifest_entries):
        motion_offsets = []
        for motion_file, manifest_entry in zip(motion_files, manifest_entries):
            if manifest_entry is None:
                motion_offsets.append(None)
                continue
            motion_offsets.append(
                _compute_manifest_motion_offset(
                    manifest_entry,
                    motion_file,
                    args_cli.motion_root_body_idx,
                    args_cli.terrain_use_manifest_pose,
                )
            )
            print(f"[INFO]: Manifest trajectory: {manifest_entry.get('trajectory_name', '')}")
        env_cfg.commands.motion.motion_position_offsets = motion_offsets
        env_cfg.commands.motion.motion_position_offset = motion_offsets[0]

    if args_cli.terrain_file is not None:
        # 当前环境每次运行只支持一个 terrain USD。多轨迹时使用第一个匹配 manifest 条目的 terrain pose。
        terrain_manifest_entry = next((entry for entry in manifest_entries if entry is not None), None)
        _add_manifest_terrain(env_cfg, args_cli.terrain_file, terrain_manifest_entry, args_cli.terrain_use_manifest_pose)

    if args_cli.fixed_start_frame >= 0:
        _use_fixed_motion_start(env_cfg, args_cli.fixed_start_frame)

    # 设置实验日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # 设置本次 run 的日志目录：{time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 如需录制视频，给环境包一层 RecordVideo
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果任务是多 agent，转换成单 agent 形式给 RSL-RL 使用
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装成 RSL-RL 需要的 VecEnv 接口
    env = RslRlVecEnvWrapper(env)

    # 创建 RSL-RL runner
    runner = StudentDaggerOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # 把 git 状态写入日志，方便之后追踪训练代码版本
    runner.add_git_repo_to_log(__file__)
    # 如果是续训，先找到旧 checkpoint 路径
    if agent_cfg.resume:
        # 获取旧 checkpoint 路径
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # 加载已经训练过的模型
        runner.load(resume_path)

    # 把环境和 agent 配置保存到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 开始训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 关闭仿真环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭 Isaac Sim app
    simulation_app.close()
