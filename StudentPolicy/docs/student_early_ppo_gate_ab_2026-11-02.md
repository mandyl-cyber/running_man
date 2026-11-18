# Student 早期 PPO 降权 A/B（2026-11-02）

## 假设与控制

此前配对 rollout 中，动作微差从早期接近阶段积累；但把前 100 参考帧锚定到原 Student，虽然固定观测动作漂移下降约 65%，闭环仍为 0/40。本实验进一步测试：早期 PPO 梯度是否直接破坏已掌握的接近动作。两组同从最佳 `model_30098.pt` 出发，256 env、训练 seed 42、50 次迭代、相同数据与优化设置，并都加载同一份重建 gate 状态后清空，以保持与上轮对照组一致。唯一变量是在 B 组把参考帧 0–99 的 PPO gate 乘 **0.1**；DAgger 和其他帧不变。A 组是上一轮的 `control_discard`，不是重新训练的第三组。

预检第 1 次 `Loss/ppo_gate_mean`：A 为 1.000，B 为 0.889，符合前缀帧占比约 12% 的预期；正式训练第 50 次 B 为 0.977，说明当时前缀帧约只占 2.6%。两组 25/50 次 checkpoint 已保存，评估均为 clean、frame-0、seed/障碍 seed 43、40 clips×1、无 Teacher 接管，报告完整。

| 指标 | A：普通早期 PPO | B：0–99 帧 PPO×0.1 |
|---|---:|---:|
| 闭环成功 | 2/40（clip 14、15） | 2/40（clip 5、20） |
| 原 Student 状态、精确对齐深度、0–99 帧动作漂移 L2 | 0.158 | 0.154 |
| 同上，100–199 帧 | 0.204 | 0.210 |
| 同上，≥200 帧 | 0.247 | 0.269 |

固定状态审计在 40 条原 Student trace 上完成，原 checkpoint 动作重放误差低于 `1e-3`。B 的前 100 帧漂移只比 A 低约 **2.7%**，后续帧反而更高；原 checkpoint 短评估为 3/40（clip 3、32、38），A/B 都未保住这些成功轨迹。

**决策：STOP。** 当前小幅早期 PPO 降权既未显著守住早期动作，也未改善闭环；不扩大到长训或多 seed。可支持的结论是：只在采样中很少出现的前 100 帧调低 PPO gate，无法阻止其他阶段的共享网络更新改变早期动作；但不能据此断言 PPO 本身无关。下一轮若继续研究动作保持，必须先改变前缀状态的训练覆盖，并同步测越障过渡的恢复，而不是继续调这一 PPO 系数。优先保留最佳 `model_30098.pt`，不要采用本轮 checkpoint。

产物：[A checkpoint](../reports/php_student/gate_rebuild_ab_v1/control_discard/baseline/model_step_050.pt)、[B checkpoint](../reports/php_student/early_ppo_ab_v1/treatment_scale01/baseline/model_step_050.pt)、[A 评估](../reports/php_student/gate_rebuild_ab_v1/eval/control_seed43/full_clip_seed43.json)、[B 评估](../reports/php_student/early_ppo_ab_v1/eval/scale01_seed43/full_clip_seed43.json)、[动作漂移审计](../reports/php_student/early_ppo_ab_v1/action_drift_vs_control.json)。
