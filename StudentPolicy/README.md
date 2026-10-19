# StudentPolicy

> 官方 PHP Teacher（Holosoma checkpoint）请使用
> `scripts/php_student/run_student.sh`。原 `scripts/rsl_rl/train_student.py`
> 仅用于历史 RSL-RL Teacher，二者的 observation/checkpoint contract 不兼容。

`StudentPolicy` 是学生端实验仓库。当前主路径从这里审计数据和 checkpoint、记录实验并启动官方 PHP/Holosoma 的 depth Student 蒸馏；旧 IsaacLab/RSL-RL 实现仅用于复现早期本地 Teacher 实验，不能加载官方 Teacher。

## 当前结构

```text
StudentPolicy/
  scripts/rsl_rl/
    train_student.py      # 学生端训练入口
    dataset_args.py       # 读取 new_teacherpolicy/data 的结构化数据集
    cli_args.py
  source/student_policy/student_policy/
    tasks.py              # 注册 Tracking-Flat-G1-Student-v0
    env_cfg.py            # 学生环境、depth camera、teacher obs group
    observations.py       # depth 图噪声、延迟、dropout、2D command
    depth_actor_critic.py # 3-layer CNN + MLP actor-critic
    rsl_rl_ppo_cfg.py     # PPO + DAgger 超参数
    dagger_runner.py      # 冻结 teacher 并执行 DAgger loss
```

教师端完成版在：

```text
new_teacherpolicy/
  scripts/rsl_rl/train.py
  source/whole_body_tracking/whole_body_tracking/
  data/<dataset>/tracking/*.npz
  data/<dataset>/tracking/batch_manifest.json
  data/<dataset>/terrain/*.usd
```

## 与 PDF 要求的对应关系

- 输入：student policy observation 保留 proprioception、上一帧 action、2D velocity command，并用 depth image 替换 teacher 的 privileged height scan。
- 深度图：`58 x 87`，ray-caster camera 更新频率 `30 Hz`，最大距离 `5 m`。
- 深度扰动：随机深度 offset `+-3 cm`、高斯噪声 `3 cm`、pixel dropout、patch dropout、图像平移和 3-4 policy step 延迟。
- 网络：actor MLP `[2048, 1024, 512, 256, 128]`，critic MLP `[512, 256, 128]`，depth backbone 为 3-layer CNN + global average pooling，输出 `32-D` depth feature。
- 训练：PPO + DAgger，DAgger loss 使用 MSE，系数 `10.0`，curriculum `10000` iterations；当 student 进入 teacher 原始终止条件外但仍在 relaxed termination 内时，mask 掉 DAgger loss。

## 安装

先激活 Isaac Lab / Isaac Sim 环境，然后在仓库根目录执行：

```bash
python -m pip install -e new_teacherpolicy/source/whole_body_tracking
python -m pip install -e StudentPolicy/source/student_policy
```

`train_student.py` 也会把上面两个 source 目录加入 `sys.path`，方便直接从当前仓库运行。

## 训练学生策略

### 快捷入口：多轨迹训练

```bash
cd /home/sjtu/running_man/StudentPolicy

python scripts/rsl_rl/train_student.py \
  --multi climb_15_z_scale_1.0 \
  --teacher_checkpoint ../new_teacherpolicy/logs/rsl_rl/g1_flat/<teacher_run>/model_20000.pt \
  --run_name climb_15_student_multi
```

`--multi DATASET` 会自动读取：

```text
../new_teacherpolicy/data/DATASET/tracking/*.npz
../new_teacherpolicy/data/DATASET/tracking/batch_manifest.json
../new_teacherpolicy/data/DATASET/terrain/*.usd
```

并自动启用 manifest 中的 terrain pose 对齐。默认补上：

```text
--task Tracking-Flat-G1-Student-v0
--max_iterations 30000
--headless
--logger tensorboard
```

### 快捷入口：单轨迹调试

```bash
python scripts/rsl_rl/train_student.py \
  --simple climb_15_z_scale_1.0 \
  --teacher_checkpoint ../new_teacherpolicy/logs/rsl_rl/g1_flat/<teacher_run>/model_20000.pt \
  --num_envs 1024 \
  --run_name climb_15_student_simple
```

`--simple` 只取该数据集 `tracking/` 下排序后的第一个 `.npz`，适合先确认 teacher checkpoint、manifest 对齐和 depth sensor 是否正常。

### 显式指定数据集

```bash
python scripts/rsl_rl/train_student.py \
  --task Tracking-Flat-G1-Student-v0 \
  --dataset_dir ../new_teacherpolicy/data \
  --dataset_name climb_15_z_scale_1.0 \
  --teacher_checkpoint ../new_teacherpolicy/logs/rsl_rl/g1_flat/<teacher_run>/model_20000.pt \
  --headless \
  --logger tensorboard \
  --run_name climb_15_student \
  --max_iterations 30000
```

也可以完全手动传入 motion、manifest 和 terrain：

```bash
python scripts/rsl_rl/train_student.py \
  --motion_files ../new_teacherpolicy/data/climb_15_z_scale_1.0/tracking/*.npz \
  --manifest_file ../new_teacherpolicy/data/climb_15_z_scale_1.0/tracking/batch_manifest.json \
  --terrain_file ../new_teacherpolicy/data/climb_15_z_scale_1.0/terrain/multi_boxes_z_scale_1.0.usd \
  --terrain_use_manifest_pose \
  --teacher_checkpoint ../new_teacherpolicy/logs/rsl_rl/g1_flat/<teacher_run>/model_20000.pt \
  --headless \
  --logger tensorboard \
  --run_name climb_15_student_manual \
  --max_iterations 30000
```

## 常用参数

- `--teacher_checkpoint`: 必填，教师端 checkpoint，必须来自 `Tracking-Flat-G1-v0` 对应 teacher observation 维度。
- `--num_envs`: 覆盖默认并行环境数，显存不够时先降到 `1024` 或 `2048` 验证流程。
- `--dataset_motion_index`: 使用结构化数据集时指定第 N 条轨迹。
- `--command_x_min / --command_x_max`: student 的前向速度命令范围，默认固定 `1.0 m/s`。
- `--command_y_min / --command_y_max`: student 的横向速度命令范围，默认固定 `0.0 m/s`。
- `--fixed_start_frame`: 调试用，强制每次 reset 从同一参考帧开始；正常训练保持 `-1`。

## 日志

学生端日志默认保存到：

```text
StudentPolicy/logs/rsl_rl/g1_flat_student/<timestamp>_<run_name>/
```

查看 TensorBoard：

```bash
cd /home/sjtu/running_man/StudentPolicy
tensorboard --logdir logs/rsl_rl/g1_flat_student
```
