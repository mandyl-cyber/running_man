# Student 续训回退定位（2026-11-01）

## 目的与控制变量

从当前最好且未被本实验覆盖的 `model_30098.pt` 出发，验证先前“普通继续训练 100 次后 clean full-clip 从 39/480 变为 0/480”的发生位置。使用相同保存配置（256 env、训练 seed 42、官方 PHP/Holosoma Student trainer），先零更新加载重存，再在普通续训的第 1、5、10、25、50 次更新保存 checkpoint。每个 checkpoint 用同一 clean、frame-0、无 Teacher 接管、seed/障碍 seed 43、40 轨迹各 1 次评估。原配置与续训配置除 `save_interval` 和自动生成的临时资产路径外一致。

## 结果

| checkpoint | 成功/40 | 解释 |
|---|---:|---|
| 原始 source | 3 | 对照 |
| 零更新加载重存 | 3 | 成功轨迹身份也完全一致；加载/保存本身未造成观测到的回退 |
| 更新 1 / 5 / 10 / 25 | 3 / 3 / 3 / 3 | 总数不变，但成功轨迹身份已变化，策略从早期就在漂移 |
| 更新 50 | **0** | 在已测里程碑中首次观测到总数下降；真实变化点只可限定在第 26–50 次之间 |

上一轮普通续训 100 次的更完整评估为 **0/480**，相同起点为 **39/480**；故 50 次短评估并非唯一回退证据。但单 seed、每轨迹 1 次的扫描不能精确判定哪一更新步使概率下降。

## 已发现的状态缺口与因果反证

- 原训练结束时 gate 已启用：`Env/ppo_gate_engaged=1`、warmup=1、`Loss/ppo_gate_mean≈0.49`。原 checkpoint 包含模型、优化器、迭代、学习率等，但没有 `env_state`，也没有 gate EMA 表与计数。
- 续训 gate 从零开始：第 50 次 warmup 仅 **0.452**，`engaged=0`，`Loss/ppo_gate_mean=1.0`。代码规定累计 50 万个有效样本前不启用相位 gate；这段续训实际使用未分相位的全量 PPO gate。这是可复现的流程缺口，不是推测。
- 但将冷启动期间 gate **单独固定为 0.5**，相同 50 次续训后仍为 **0/40**；日志确认 PPO gate 均值确实为 0.5。因此该缺口**不足以单独解释或修复**回退。固定 0.5 也不等价于恢复原有逐相位 gate 表。
- 两组续训学习率记录为 `1e−5`，PPO 系数 `0.9`；普通续训第 1/25/50 次的 KL 约 `0.079/0.044/0.046`，高于配置 `desired_kl=0.01`，但原训练末期 KL 也约 `0.05`，不能据此单独归因。训练回报上升同时 clean 成功率下降，提示训练目标与验收目标仍不一致。

## 结论与下一步

**不继续加大 DAgger 权重，也不直接继续训练到更久。** 现已排除“仅加载重存即坏”的解释，并确认第 26–50 次之间出现观测到的整体回退；gate 状态未恢复是真实问题，但半权重对照未救回策略。下一步应在同一 25→50 窗口记录每轨迹的失败帧、Teacher 有效标签/动作误差、窗口 PPO/DAgger 梯度与 depth backbone 参数变化；再做“零参数更新地先收集足够残差、重建逐相位 gate 表”与原续训的单变量 A/B，而不是把均值 0.5 当作原表替身。若重建表仍失败，再检验 PPO 更新及训练采样分布。

结果：[sweep_summary.json](../reports/php_student/continuation_collapse_v1/sweep_summary.json)。各 checkpoint 与逐轨迹 JSON 均保留在同目录；诊断入口为 [run_student_phase_ab.sh](../scripts/php_student/run_student_phase_ab.sh) 和 [run_student_continuation_sweep_eval.sh](../scripts/php_student/run_student_continuation_sweep_eval.sh)。
