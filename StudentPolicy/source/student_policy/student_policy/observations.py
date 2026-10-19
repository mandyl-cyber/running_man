from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    max_distance: float = 5.0,
    offset_range: tuple[float, float] = (-0.03, 0.03),
    noise_std: float = 0.03,
    delay_steps_range: tuple[int, int] = (3, 4),
    dropout_prob: float = 0.01,
    patch_dropout_prob: float = 0.05,
    patch_size_range: tuple[int, int] = (4, 12),
    pixel_shift_range: tuple[int, int] = (-2, 2),
    focal_scale_range: tuple[float, float] = (0.96, 1.04),
) -> torch.Tensor:
    """Return flattened normalized depth images for the student policy.

    The raw ray-caster camera output has shape ``(num_envs, H, W, 1)``. We clip
    invalid values, add the depth artifacts described in the paper appendix, and
    normalize to approximately ``[0, 1]`` before flattening for RSL-RL storage.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type].squeeze(-1)
    depth = _apply_camera_extrinsic_jitter(env, depth, pixel_shift_range, focal_scale_range)
    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    depth = torch.clamp(depth, 0.0, max_distance)

    if offset_range != (0.0, 0.0):
        offset = torch.empty((depth.shape[0], 1, 1), device=depth.device).uniform_(*offset_range)
        depth = depth + offset
    if noise_std > 0.0:
        depth = depth + torch.randn_like(depth) * noise_std

    depth = torch.clamp(depth, 0.0, max_distance) / max_distance
    depth = _apply_depth_dropout(depth, dropout_prob, patch_dropout_prob, patch_size_range)
    depth = _apply_depth_delay(env, depth, delay_steps_range)
    return depth.reshape(depth.shape[0], -1)


def velocity_command(
    env: ManagerBasedEnv,
    x_range: tuple[float, float] = (1.0, 1.0),
    y_range: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Return an explicit 2-D command sampled per environment at reset."""

    device = env.scene.env_origins.device
    if not hasattr(env, "_student_velocity_command"):
        env._student_velocity_command = torch.zeros(env.num_envs, 2, device=device)
        _resample_velocity_command(env, torch.arange(env.num_envs, device=device), x_range, y_range)

    reset_ids = torch.where(env.episode_length_buf <= 1)[0]
    if len(reset_ids) > 0:
        _resample_velocity_command(env, reset_ids, x_range, y_range)

    return env._student_velocity_command


def _resample_velocity_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
):
    command = env._student_velocity_command
    command[env_ids, 0] = torch.empty(len(env_ids), device=command.device).uniform_(*x_range)
    command[env_ids, 1] = torch.empty(len(env_ids), device=command.device).uniform_(*y_range)


def _apply_camera_extrinsic_jitter(
    env: ManagerBasedEnv,
    depth: torch.Tensor,
    pixel_shift_range: tuple[int, int],
    focal_scale_range: tuple[float, float],
) -> torch.Tensor:
    """Approximate small camera mounting errors with per-env image-plane warps."""

    if pixel_shift_range == (0, 0) and focal_scale_range == (1.0, 1.0):
        return depth

    device = depth.device
    num_envs, height, width = depth.shape
    if (
        not hasattr(env, "_student_camera_jitter")
        or env._student_camera_jitter["shape"] != (num_envs, height, width)
        or env._student_camera_jitter["device"] != device
    ):
        env._student_camera_jitter = {
            "shape": (num_envs, height, width),
            "device": device,
            "dx": torch.zeros(num_envs, device=device),
            "dy": torch.zeros(num_envs, device=device),
            "scale": torch.ones(num_envs, device=device),
        }
        _resample_camera_jitter(env, torch.arange(num_envs, device=device), pixel_shift_range, focal_scale_range)

    reset_ids = torch.where(env.episode_length_buf <= 1)[0]
    if len(reset_ids) > 0:
        _resample_camera_jitter(env, reset_ids, pixel_shift_range, focal_scale_range)

    jitter = env._student_camera_jitter
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(num_envs, 1, 1, 1)
    scale = jitter["scale"].view(num_envs, 1, 1, 1)
    shift_x = (2.0 * jitter["dx"] / max(width - 1, 1)).view(num_envs, 1, 1)
    shift_y = (2.0 * jitter["dy"] / max(height - 1, 1)).view(num_envs, 1, 1)
    grid[..., 0] = grid[..., 0] / scale.squeeze(-1) - shift_x
    grid[..., 1] = grid[..., 1] / scale.squeeze(-1) - shift_y

    warped = F.grid_sample(depth.unsqueeze(1), grid, mode="nearest", padding_mode="border", align_corners=True)
    return warped.squeeze(1)


def _resample_camera_jitter(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pixel_shift_range: tuple[int, int],
    focal_scale_range: tuple[float, float],
):
    jitter = env._student_camera_jitter
    low, high = pixel_shift_range
    jitter["dx"][env_ids] = torch.randint(low, high + 1, (len(env_ids),), device=jitter["device"]).float()
    jitter["dy"][env_ids] = torch.randint(low, high + 1, (len(env_ids),), device=jitter["device"]).float()
    jitter["scale"][env_ids] = torch.empty(len(env_ids), device=jitter["device"]).uniform_(*focal_scale_range)


def _apply_depth_dropout(
    depth: torch.Tensor,
    dropout_prob: float,
    patch_dropout_prob: float,
    patch_size_range: tuple[int, int],
) -> torch.Tensor:
    if dropout_prob > 0.0:
        mask = torch.rand_like(depth) < dropout_prob
        depth = depth.masked_fill(mask, 1.0)

    if patch_dropout_prob <= 0.0:
        return depth

    h, w = depth.shape[-2:]
    patch_size = max(1, int((patch_size_range[0] + patch_size_range[1]) * 0.5))
    coarse_h = max(1, (h + patch_size - 1) // patch_size)
    coarse_w = max(1, (w + patch_size - 1) // patch_size)
    coarse_mask = torch.rand(depth.shape[0], 1, coarse_h, coarse_w, device=depth.device) < patch_dropout_prob
    mask = F.interpolate(coarse_mask.float(), size=(h, w), mode="nearest").squeeze(1).bool()
    return depth.masked_fill(mask, 1.0)


def _apply_depth_delay(
    env: ManagerBasedEnv,
    depth: torch.Tensor,
    delay_steps_range: tuple[int, int],
) -> torch.Tensor:
    min_delay, max_delay = delay_steps_range
    if max_delay <= 0:
        return depth

    device = depth.device
    buffer_len = max_delay + 1
    if (
        not hasattr(env, "_student_depth_delay_buffer")
        or env._student_depth_delay_buffer.shape[0] != buffer_len
        or env._student_depth_delay_buffer.shape[1:] != depth.shape
    ):
        env._student_depth_delay_buffer = depth.unsqueeze(0).repeat(buffer_len, 1, 1, 1)
        env._student_depth_delay_steps = torch.full((depth.shape[0],), max_delay, dtype=torch.long, device=device)
        env._student_depth_delay_write_index = 0

    reset_ids = torch.where(env.episode_length_buf <= 1)[0]
    if len(reset_ids) > 0:
        env._student_depth_delay_steps[reset_ids] = torch.randint(
            min_delay, max_delay + 1, (len(reset_ids),), dtype=torch.long, device=device
        )
        env._student_depth_delay_buffer[:, reset_ids] = depth[reset_ids].unsqueeze(0)

    write_index = env._student_depth_delay_write_index
    env._student_depth_delay_buffer[write_index] = depth
    delayed_indices = (write_index - env._student_depth_delay_steps) % buffer_len
    env_ids = torch.arange(depth.shape[0], device=device)
    delayed = env._student_depth_delay_buffer[delayed_indices, env_ids]
    env._student_depth_delay_write_index = (write_index + 1) % buffer_len
    return delayed
