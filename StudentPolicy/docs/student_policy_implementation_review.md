# Student Policy 实现与训练注意事项

> 历史文档：本文描述旧 IsaacLab/RSL-RL 实现，不代表当前官方 PHP/Holosoma
> Student 合同。正式对齐结论见 `official_teacher_student_alignment_2026-10-19.md`；
> 官方实现使用 15 维 motion one-hot command，而不是本文旧实现的固定 2 维 command。

本文基于 `new_teacherpolicy/whole_body_tracking_cp.pdf` 中的 student distillation 目标，对当前 `StudentPolicy` 实现进行说明，并指出和 PDF 目标相比需要注意的问题。

## PDF 中的 Student 目标

PDF 的 student 阶段位于所有 per-skill experts 训练完成之后，目标是把多个 expert 蒸馏成一个 visuomotor student policy。

核心设定：

- **输入**：proprioception、depth image、2D velocity command、previous action。
- **深度图**：`58 x 87` resolution。
- **网络**：3-layer CNN + Global Average Pooling，depth feature dim 为 `32`。
- **Actor MLP**：`[2048, 1024, 512, 256, 128]`。
- **Critic MLP**：`[512, 256, 128]`。
- **Initial noise std**：`0.01`。
- **训练目标**：DAgger + PPO hybrid distillation。
- **DAgger curriculum**：训练前半段逐步降低 DAgger 权重，但不要降到 0，最低保留 `0.1`。
- **为什么不能只用 DAgger**：攀爬、跨越等技能有短时间高力矩关键动作，单步 imitation loss 不足以表达 episode-level success。
- **Best practice**：student 失败但 expert 成功时，不要只靠 DAgger；保持 PPO，放宽 student termination，并保持 DAgger 正则。

## 当前实现概览

当前 student 代码主要位于：

- `StudentPolicy/source/student_policy/student_policy/env_cfg.py`
- `StudentPolicy/source/student_policy/student_policy/observations.py`
- `StudentPolicy/source/student_policy/student_policy/depth_actor_critic.py`
- `StudentPolicy/source/student_policy/student_policy/dagger_runner.py`
- `StudentPolicy/source/student_policy/student_policy/rsl_rl_ppo_cfg.py`
- `StudentPolicy/scripts/rsl_rl/train_student.py`

### 环境与任务注册

`Tracking-Flat-G1-Student-v0` 注册在：

```text
StudentPolicy/source/student_policy/student_policy/tasks.py
```

它使用：

```python
env_cfg.G1FlatStudentRelaxedEnvCfg
student_policy.rsl_rl_ppo_cfg:G1FlatStudentPPORunnerCfg
```

也就是说，student 默认使用 relaxed termination，`anchor_pos` threshold 被放宽到 `1.0`。这符合 PDF 对 student distillation 更宽松 termination 的建议。

### Student Actor Observation

当前 student policy observation 在 `StudentObservationsCfg.PolicyCfg` 中定义：

```python
command
projected_gravity
base_ang_vel
joint_pos
joint_vel
depth_image
actions
```

其中 `command` 在 `G1FlatStudentEnvCfg.__post_init__` 中被替换成 `student_obs.velocity_command`，输出 2D command：

```python
self.observations.policy.command.func = student_obs.velocity_command
self.observations.policy.command.params = {"x_range": COMMAND_X_RANGE, "y_range": COMMAND_Y_RANGE}
```

这和 PDF 中 student 使用 2D velocity command 的目标一致。

### Teacher Observation Group

为了 DAgger 查询 frozen teacher，当前 student 环境额外定义了 `teacher` observation group：

```python
command
motion_anchor_pos_b
motion_anchor_ori_b
base_lin_vel
base_ang_vel
joint_pos
joint_vel
actions
```

这组 observation 用来匹配 `Tracking-Flat-G1-v0` teacher actor 的输入维度和顺序。`dagger_runner.py` 中也有 checkpoint actor input dim 检查，能避免加载 observation 维度不匹配的 teacher。

### Depth Camera 与 Depth Augmentation

当前 depth camera 使用 `MultiMeshRayCasterCameraCfg`：

```python
height = 58
width = 87
max_distance = 5.0
update_period = 1.0 / 30.0
```

`observations.py` 中对 depth 做了：

- invalid value clamp；
- max distance normalize；
- depth offset；
- Gaussian noise；
- pixel dropout；
- patch dropout；
- image-plane shift；
- focal scale jitter；
- 3-4 steps delay。

这比 PDF 主文档描述更具体，方向上符合 student visuomotor policy 对深度扰动鲁棒性的需求。

### Student Network

`StudentDepthActorCritic` 实现了：

- 3-layer depth CNN；
- AdaptiveAvgPool2d；
- depth feature dim `32`；
- actor MLP `[2048, 1024, 512, 256, 128]`；
- critic MLP `[512, 256, 128]`；
- initial std `0.01`。

这和 PDF 的 student architecture 基本一致。

### DAgger + PPO Runner

`StudentDaggerOnPolicyRunner` 的训练流程是：

1. 每个 rollout step 查询 frozen teacher action。
2. 用当前 student action 与环境交互，收集 PPO rollout。
3. 先执行 `self.alg.update()` 做 PPO 更新。
4. 再执行 `_update_dagger(...)` 做一个 DAgger MSE 更新。
5. DAgger weight 从 `1.0` 线性降到 `0.1`。

当前 DAgger loss：

```python
per_env_loss = MSE(student_action_mean, teacher_actions)
loss = dagger_loss_coef * dagger_weight * dagger_loss
```

默认：

```python
dagger_loss_coef = 10.0
dagger_curriculum_iterations = 10000
```

保留非零 DAgger 权重这一点符合 PDF。

## 和 PDF 一致的部分

- student 使用 proprioception + previous action + depth image + 2D velocity command。
- depth resolution 是 `58 x 87`。
- depth backbone 是 3-layer CNN + GAP，输出 `32-D` feature。
- actor/critic hidden dims 和 PDF 对齐。
- student initial noise std 是 `0.01`。
- student 使用 PPO + DAgger，而不是纯 DAgger。
- DAgger 权重最终保留 `0.1`，没有降到 0。
- student termination 放宽到 `1.0m`，符合 PDF troubleshooting 对 student/mirrored execution 的建议。

## 当前实现中需要注意的问题

### 1. Student 多轨迹训练仍可能继承 teacher 的 terrain/reference 对齐问题

`train_student.py` 和 teacher 训练脚本一样，会对每条 motion 计算自己的 motion offset：

```python
env_cfg.commands.motion.motion_position_offsets = motion_offsets
```

但 terrain 仍然只根据第一条有效 manifest entry 加载一次：

```python
terrain_manifest_entry = next((entry for entry in manifest_entries if entry is not None), None)
_add_manifest_terrain(...)
```

如果同一个 dataset 中不同 trajectories 的 `terrain_world_pose` 不同，就会出现 reference motion 和实际 terrain collision 错位。  
这个问题不只影响 teacher，也会影响 student，因为 student 的 depth image、teacher action、reward reference 都依赖同一个环境几何关系。

#### 建议

- 在 student 多轨迹训练前，必须先修复 teacher 文档中提到的 terrain/reference paired asset 问题。
- 不要用一个错位或不稳定的 teacher checkpoint 训练 student。
- 对 student dataset 也做 `terrain_world_pose` 一致性检查。

### 2. DAgger mask 与“teacher 原始终止条件”不完全一致

README 中写的是：当 student 进入 teacher 原始终止条件外但仍在 relaxed termination 内时，mask 掉 DAgger loss。

但当前 `_expert_mask()` 使用的是：

```python
bad_anchor_pos_z_only(...)
bad_anchor_ori(...)
bad_motion_body_pos_z_only(...)
```

这里有两个偏差：

- teacher 原始 termination 主要是 full anchor position error `bad_anchor_pos(threshold=0.5)`。
- student mask 使用的是 `bad_anchor_pos_z_only`，只看 z 方向，不看 x/y tracking drift。

这会导致一种情况：student 在水平位置已经明显偏离 teacher/reference，但 z 误差还没超阈值，此时 DAgger loss 仍然有效，teacher action 会在 teacher 自己不熟悉或已经失效的状态上监督 student。

#### 建议

- 如果目标是“严格按 teacher 原始 termination mask DAgger”，应把 `bad_anchor_pos_z_only` 改成 `bad_anchor_pos`。
- 或者明确当前 mask 是一个更宽松的工程策略，并在日志中监控 full anchor position error。
- 增加 `dagger_valid_ratio`、full anchor error、z-only anchor error 的对比日志。

### 3. 当前 loss 实现不是 PDF 公式中的单个 combined objective

PDF 写的是：

```text
L = lambda_PPO * L_PPO + lambda_D * L_D
lambda_PPO + lambda_D = 1
```

当前代码是先做一次 PPO optimizer step，再单独做一次 DAgger optimizer step：

```python
loss_dict = self.alg.update()
mean_dagger_loss, dagger_weight, dagger_valid_ratio = self._update_dagger(...)
```

这不是严格的单个 combined objective。它是“PPO update + supervised update”的交替优化。工程上可以工作，但和 PDF 公式不同。

另一个差异是当前 `dagger_loss_coef = 10.0`，并没有和 PPO loss 做 `lambda_PPO + lambda_D = 1` 的归一化。DAgger 梯度可能在早期非常强，尤其当 teacher action 和 student action 差距较大时。

#### 建议

- 如果优先复现 PDF，可以考虑把 DAgger loss 合入 PPO update 的总 loss 中，或者至少让两个 loss 的权重更接近 `lambda_PPO/lambda_D` 的解释。
- 如果保留当前两步优化，需要重点监控：
  - `Loss/dagger`
  - `Loss/dagger_weight`
  - PPO surrogate loss
  - value loss
  - KL
  - action std
- 如果出现 student action 被 teacher 强拉、PPO reward 不涨，可先降低 `dagger_loss_coef`，例如从 `10.0` 测试 `1.0`、`3.0`、`5.0`。

### 4. 2D velocity command 当前和 reference motion 没有真实绑定

PDF 中 student 使用 2D velocity command，是为了让单个 visuomotor student 根据命令执行不同移动意图。

当前实现中，`velocity_command` 默认固定为：

```python
COMMAND_X_RANGE = (1.0, 1.0)
COMMAND_Y_RANGE = (0.0, 0.0)
```

并且这个 command 只是 student actor 的 observation，没有驱动 motion reference 选择，也没有直接进入 reward 目标。真实 tracking target 仍来自 `MotionCommand` 中的 reference motion。

如果后续把 command range 放宽，但不让 command 和 reference trajectory / teacher skill 对应起来，student 会看到一个和当前专家动作不一致的 command，训练会变成部分错误监督。

#### 建议

- 当前阶段保持固定 command 是合理的。
- 如果要训练多速度或多方向 student，需要让 command 与 active motion 的方向、速度或 skill choice 对齐。
- 不要简单扩大 `command_x/y_range` 后直接训练，除非 reward/reference 也按 command 条件化。

### 5. Depth slicing 依赖 observation 顺序，比较脆弱

`StudentDepthActorCritic` 假设 depth image 位于 actor observation 中倒数 `num_actions` 之前：

```python
self.depth_start = num_actor_obs - self.depth_numel - num_actions
self.depth_end = self.depth_start + self.depth_numel
```

这依赖 `PolicyCfg` 的 observation 顺序必须是：

```text
... depth_image, actions
```

当前代码满足这个顺序。但以后如果增加 observation，尤其是在 `depth_image` 后面、`actions` 前面插入内容，就会导致 CNN 读取错误切片，训练会直接异常或隐性失败。

#### 建议

- 保持 `depth_image` 紧挨 `actions` 之前。
- 在初始化时增加 assertion，检查 depth slice 的数值范围和 shape。
- 更稳妥的方案是让 observation manager 保留 term slice metadata，而不是靠固定位置推断。

### 6. Depth delay 和 velocity command 的 reset 判断依赖 observation 调用时机

`velocity_command`、camera jitter、depth delay 都使用：

```python
reset_ids = torch.where(env.episode_length_buf <= 1)[0]
```

这通常可用，但 observation function 可能在 reset 后、step 前、logging 或 teacher/student 多组 observation 计算时被多次调用。默认 velocity command 是固定 `(1.0, 0.0)`，所以影响不明显；如果未来 command range 随机化，可能在 episode 开头被多次重采样。

#### 建议

- 固定 command 阶段可以保留。
- 若使用随机 command，最好把 command resampling 放到 reset event 或 command manager 中，而不是 observation function 内。
- 对 depth delay buffer，在 reset 后确保 buffer 不混入上一个 episode 的图像。

### 7. Teacher checkpoint 质量是 student 上限

PDF 的 student 阶段假设 experts 已经能完成对应 skill。  
如果 teacher 在 24 tracks 上本身 collapse 或卡在障碍边缘，student 通过 DAgger 会学习失败动作，PPO 也很难弥补。

#### 建议

- 只使用已经通过 play/replay 验证的 teacher checkpoint。
- 每个 teacher checkpoint 至少验证：
  - 单轨迹成功率；
  - 多起点 reset 成功率；
  - 障碍边缘跨越成功率；
  - terrain/reference 对齐正确；
  - teacher observation 维度与 student `TeacherCfg` 匹配。

## 训练注意事项

### 1. 先单轨迹 smoke test，再多轨迹

建议顺序：

```text
single trajectory + fixed_start_frame
single trajectory + random start frame
3 trajectories
更多 trajectories
完整 dataset
```

每一步确认 reward、episode length、DAgger loss、depth image 都正常后再扩大。

### 2. 先确认 depth camera 真的看到了 terrain

student 和 teacher 不同，student actor 主要依赖 depth 来感知障碍。训练前必须确认：

- `depth_camera.mesh_prim_paths` 指向 `{ENV_REGEX_NS}/Terrain`。
- depth 图中能看到障碍，而不是全是最大距离或默认平面。
- camera pose 没有被 torso link 遮挡。
- depth max distance、near obstacle geometry、resolution 合理。

### 3. 监控 DAgger 和 PPO 是否互相冲突

重点看：

- DAgger loss 是否持续下降。
- PPO reward 是否同步上升。
- 如果 DAgger loss 下降但 reward 不升，可能是在模仿失败 teacher 或错误状态。
- 如果 reward 上升但 DAgger loss 很高，可能是 student 找到和 teacher 不同但成功的动作；这时 DAgger 权重太高可能反而限制学习。

### 4. 不要过早扩大 command range

当前 command 固定为 `(1.0, 0.0)` 更适合先复现基础 pipeline。  
多 command 训练应等到：

- 单 command student 已经稳定；
- reference/motion 选择能和 command 对齐；
- reward 能表达 command-conditioned behavior。

### 5. DAgger valid ratio 很重要

`Loss/dagger_valid_ratio` 如果长期很低，说明大部分状态都被 mask 掉，teacher supervision 很少。  
如果长期接近 1 但 reward 很差，要检查 mask 是否过宽，是否在 teacher 不可靠状态上继续 imitation。

### 6. Student 不应早于 teacher 修复

如果 teacher 的 terrain/reference 对齐、per-skill expert、critic privileged observation 没修好，student 训练结果很难可靠。  
建议先完成 teacher 侧：

- 单 skill expert 稳定；
- terrain/reference paired asset 正确；
- checkpoint play 成功；
- teacher observation 与 student `TeacherCfg` 完全匹配。

## 推荐修复优先级

### P0：先保证 teacher checkpoint 正确

student 训练前，teacher 必须已经能稳定完成对应 skill。  
不要用 collapse 的 24 tracks teacher checkpoint 训练 student。

### P1：修复 multi-trajectory terrain/reference 对齐

student 的 `train_student.py` 和 teacher 一样存在“多 motion offsets + 单 terrain pose”的风险。  
如果 dataset 中每条 trajectory 的 `terrain_world_pose` 不同，必须先统一坐标或拆分训练。

### P2：修正或明确 DAgger mask

如果文档目标是“teacher 原始 termination 外 mask DAgger”，代码应使用 full anchor position error，而不是只使用 z-only。

### P3：检查 combined loss 设计

当前两步优化不是 PDF 公式的严格实现。  
如果训练不稳定，优先调 `dagger_loss_coef` 和 DAgger curriculum，必要时再把 DAgger 合入 PPO update。

### P4：保护 depth observation slicing

保持 `depth_image` 在 `actions` 前面，并增加 shape/range 检查，避免后续改 observation 时静默破坏 CNN 输入。

### P5：逐步扩大数据和 command 难度

先固定 command、少量轨迹、可靠 teacher；再扩大 tracks、command range、domain randomization。

## 验证 Checklist

训练 student 前至少确认：

- teacher checkpoint 已经能完成目标 skill。
- teacher checkpoint 的 actor input dim 和 `StudentObservationsCfg.TeacherCfg` 匹配。
- student dataset 的 terrain/reference 对齐正确。
- depth camera 输出不是空图、不是全最大距离。
- student policy observation 中 depth slice 位置正确。
- DAgger mask 和预期 teacher termination 一致。
- `Loss/dagger_valid_ratio` 不长期为 0 或异常接近 1。
- fixed command 阶段稳定后，再考虑 command randomization。
- student reward 上升和 DAgger loss 下降趋势不互相矛盾。

