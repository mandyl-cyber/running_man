# Student 训练信号采样审计（2026-10-30）

从当前最佳短实验 `model_30098.pt` 加载保存训练配置，512 env、不更新参数，采 70×24 步、共 **860,160** 条真实 DistillationPPO storage transition。采样前 Teacher action 与 storage 标签最大误差为 **0**。此实验重新初始化训练环境与 gate 统计，**不是原始 30k 训练历史的重放**；motion 池也不直接等同于 clean 评估的 40 条 clip。

| reference 相位 | 样本 | 有效 Teacher 标签 | 有效标签上的 Student–Teacher action L2 | 平均有效 PPO 系数 | 平均有效 DAgger 权重* |
|---|---:|---:|---:|---:|---:|
| 0–20% | 64,052 | 98.9% | 0.791 | 0.807 | 1.905 |
| 20–40% | 171,967 | 62.9% | 1.441 | 0.870 | 0.777 |
| 40–60% | 196,475 | 62.4% | 1.188 | 0.835 | 0.998 |
| 60–80% | 197,661 | 83.6% | 1.291 | 0.827 | 1.449 |
| 80–100% | 230,005 | 77.0% | 0.797 | 0.808 | 1.490 |

`*` DAgger 权重按每条 transition 的 `(1 − ppo_coef × gate) × dagger_loss_coef × valid_label` 计算；无效标签权重为 0。此处 checkpoint 的 `ppo_coef=0.9`、`dagger_loss_coef=10`。在 frame 180–250 聚合样本中，66,777 条 transition 的有效标签率 **89.1%**、有效动作 L2 **1.381**、平均有效 PPO 系数 **0.886**、平均有效 DAgger 权重 **1.015**。

运行末 gate 已启动：累计 636,359 个有效标签，超过 500,000 门槛；**但表中均值混合了 gate 预热前后**，不得解释为纯 gate 启动后的权重。45-rollout 先导采样只有 409,162 个有效标签，gate 尚未启动，所有 gate 值为 1；因此它只用于确认预热语义。

## 结论与限制

- 目前不支持“关键过渡阶段 PPO gate 被整体关闭”这一简单解释：frame 180–250 聚合 PPO 系数接近 0.9。**系数非零不等于有效 PPO 梯度或成功奖励已传到关键动作**，这部分尚未测量。
- 20–60% 相位同时存在更高动作误差和约 38% 无效 Teacher 标签。这与 Student 闭环偏差/监督缺口一致，但不能由此唯一归因于标签覆盖。
- frame 180–250 是跨训练池 motion 的绝对帧聚合，不等同于 clean 评估中 clip 4/12 的同一接触事件。下一轮应在训练 storage 中按 motion 名称、reference 帧和接触事件分层，并在 gate 生效后单独统计梯度/优势。先不修改 PPO 系数或延长训练。

代码：`scripts/php_student/probe_student_storage_coverage.py`；数据：`reports/php_student/storage_weights_best_512env_45rollouts_20261030.json`、`storage_weights_best_512env_70rollouts_20261030.json`。

## 仅 gate 启动后的复核

在相同 512 env×70 rollouts 的无更新复采中，逐步标记 gate 是否实际启动，再单独汇总启动后的 **179,200** 条 transition：整体有效标签率 **73.2%**、平均有效 PPO 系数 **0.569**；frame 180–250 的 **13,857** 条 transition 中有效标签率 **87.8%**、有效标签上的动作 L2 **1.412**、平均有效 PPO 系数 **0.828**。这说明 gate 在该聚合帧段并未关闭 PPO，反而相对整体提高了权重；但**没有测量该段 PPO advantage、梯度或成功回报是否有效**。

训练环境加载的是 **120 个子 motion**，当前预处理将它们拼成一个临时 NPZ；采样器只暴露这个合并路径。因此训练 motion ID 4/12/24 **不能可靠映射**到 clean 评估 clip 4/12/24，任何逐 ID 对照都应暂停，先补原始名称映射。该复核数据：`reports/php_student/storage_weights_best_postgate_512env_70rollouts_20261030.json`。
