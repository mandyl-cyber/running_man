# StudentPolicy 对齐官方 PHP：修改与技术审计

日期：2026-10-19

## 1. 目标与结论

目标是使用已通过 40 条 motion clean evaluation 的官方 PHP Teacher `model_19999.pt`，在本地 `StudentPolicy` 仓库管理 Student 蒸馏实验。

结论：旧版 Student 不是空实现，但它是 IsaacLab/RSL-RL 接口，不能安全加载 Holosoma Teacher。当前采用两条明确隔离的路径：

- **正式路径**：本仓库负责 checkpoint/data/config 审计和启动；训练计算使用锁定版本的官方 PHP/Holosoma 实现。
- **历史路径**：保留旧 RSL-RL Student，只服务于旧版 `new_teacherpolicy` Teacher，不宣称与官方 Student 等价。

这不是简单的 checkpoint 键名问题。官方 Teacher 的 449 维 observation 含义和排序、动作缩放、terrain preprocess、motion routing 都由 Holosoma 环境定义。直接复制权重即使维度能对上，也可能让输入语义错位。

## 2. 官方合同基线

锁定源码版本：

- PHP：`c7c9ce3fc75a8754a7e403c58de3b6933254d01b`
- Holosoma：`4a0bf04cde5930af5ed59da797568d32be972525`

| 项目 | 官方配置 |
|---|---|
| Student actor observation | proprioception、上一动作、15 维 motion velocity one-hot |
| Teacher observation | 原 Teacher 449 维 actor observation + 动态定位的 `which_motion` 路由列 |
| Critic observation | privileged state + terrain height scan |
| Depth | D435i Warp ray caster，58×87，3–4 帧延迟，buffer 6，holes 开启 |
| Depth encoder | `DepthOnlyFCBackbone58x87Small`，输出 32 维 |
| Student / Teacher / Critic MLP | `[2048,1024,512,256,128]` / `[512,256,128]` / `[512,256,128]` |
| 优化 | PPO + DAgger combined update；MSE，DAgger 系数 10 |
| PPO | 24 steps，2 epochs，96 minibatches，lr 3e-4，KL 0.01，entropy 0.001 |
| 训练预算 | 2048 env，30000 iterations，save interval 500 |
| schedule | PPO 从 iteration 0 渐增；DAgger curriculum 到 10000 |
| motion sampling | 关闭 adaptive timestep sampler，时间上均匀采样 |
| action | scale 1.0，不使用 effort/kp 二次缩放 |
| terrain | 每条 reference 预处理 3 个 terrain variants |
| termination curriculum | step 24000–240000 将 tracking threshold 线性放宽至 2 倍 |
| expert-lost handling | `which_motion=-1`，Teacher action 置零且 DAgger loss mask |
| normalization | `empirical_normalization=False` |

## 3. 旧实现的主要偏差

| 模块 | 旧实现 | 风险 |
|---|---|---|
| checkpoint | RSL-RL `model_state_dict` | 官方使用 `actor_model_state_dict`，不能直接加载 |
| Student command | 固定 2 维连续速度 `(1,0)` | 官方是 motion 数据驱动的 15 维 one-hot，reference 条件缺失 |
| depth | IsaacLab ray caster + 本地增强 | 不等于官方 Warp D435i 的遮洞、延迟和 terrain mesh 注入 |
| optimization | PPO 后另做 DAgger optimizer step | 不等于官方 combined loss 与 residual-gated PPO |
| expert mask | pelvis position 仅检查 z | x/y 已漂移时仍会学习错误 Teacher label |
| sampler | 强制 adaptive ratio | 官方 Student 使用 uniform-in-time |
| action scale | 继承旧 Teacher 环境 | 可能与官方 noscale Teacher 不一致 |
| termination | 固定 relaxed threshold | 缺少官方动态 curriculum |
| training budget | 20000 iterations | 官方为 30000 |
| normalization | 开启 | 官方关闭 |
| action std | 裸参数，无正值保护 | 曾导致 `Normal(std<0)` 崩溃 |

因此，本次没有把旧后端改名后冒充官方后端；完整官方语义由固定版本 Holosoma 保证。

## 4. 本次代码修改

正式路径：

1. `scripts/php_student/inspect_teacher.py` 校验 checkpoint 字段、449→29 Actor 以及 motion/terrain 一一配对，并输出 JSON。
2. `scripts/php_student/validate_official_contract.py` 校验源码 revision，并检查网络、PPO、DAgger、depth、curriculum、sampler 和 routing 的关键合同；漂移时 fail closed。
3. `scripts/php_student/run_student.sh` 提供 `audit → preflight → train`，每次运行前完成双重审计，随后调用固定官方 launcher。
4. `dagger_runner.py` 检测 Holosoma checkpoint 后立即报错，阻止旧 Runner 错误加载。

历史 RSL-RL 路径的安全修复：

1. `depth_actor_critic.py` 将 policy std 下限设为 `1e-6`，修复负标准差崩溃。
2. `dagger_runner.py` 将专家有效性从 z-only pelvis error 改为完整 3D anchor position error。
3. `rsl_rl_ppo_cfg.py` 与 `train_student.py` 默认预算改为 30000，关闭 empirical normalization。

这些修复不表示旧后端已变成官方实现；2D command、depth renderer、combined loss、routing 和 terrain preprocess 仍有根本差异。

## 5. 当前审计结果

已有 `reports/php_student/teacher_bundle_audit.json`：Teacher iteration 19999，Actor 449→29，motion/terrain 40/40 配对。正式训练前还必须生成并通过 `official_contract_audit.json`，随后完成 GPU preflight。

## 6. 执行流程与 Gate

```bash
cd /home/sjtu/running_man/StudentPolicy

export TEACHER_CHECKPOINT=/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261015_072805-php-teacher-seed42-locomotion/model_19999.pt
export PHP_BUNDLE=/home/sjtu/.cache/php-parkour/motions-low-climb-76

bash scripts/php_student/run_student.sh audit
bash scripts/php_student/run_student.sh preflight
bash scripts/php_student/run_student.sh train
```

- Gate A：两个 JSON 均 `compatible=true`，revision 命中锁定版本，40/40 配对。
- Gate B：5-iteration preflight 完成 Teacher load、rollout 和至少一次 update；shape 正确；loss/action/std 无 NaN/Inf；产生 checkpoint。
- Gate C：正式训练中 behavior loss 有下降趋势，expert-valid coverage 不长期接近 0，episode length/reward 不持续坍塌，checkpoint 与 ONNX 正常生成。

## 7. 边界与后续

- 入口和审计记录位于本仓库；官方兼容 checkpoint、日志及 ONNX 默认位于 `/home/sjtu/php_parkour_official/logs/WBT-Holosoma`。
- `PHP_ALLOW_UNPINNED=1` 只能用于有意升级版本后的重新审计，不能日常绕过。
- preflight 通过后再启动 30k。完成后用独立 seeds 做 clean、reset/model/push/robust 评估，并报告 per-motion worst case，不能只报告训练均值。
- 当前 `StudentPolicy` 不是 git repository；修改已落盘并由本文档记录，但尚无本仓库 commit 历史。
