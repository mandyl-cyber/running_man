# Student 多事件恢复训练 A/B（2026-10-29）

## 先验诊断

从提前 roll-in checkpoint `model_30098.pt` 的无接管 seed43 trace 出发，在首次持续动作分歧点让 Teacher 控制 80 步后交还 Student，14/40 完成，26 条失败都发生在交还之后。失败轨迹交还后再出现持续 Student–Teacher 动作分歧的中位延迟为 29 帧；之后 Teacher 标签逐渐失效。单次接管后，失败到来中位延迟为 239 帧，故不能把它说成“交还即崩溃”。

诊断性重复接管（最多 3 次×80 步）在无冷却时为 36/40，但平均 Teacher 控制约 234 步，接近长时间辅助，不能当作 Student 成绩。加 40 步冷却并记录事件帧后，辅助完成 **21/40**，对无接管基线 **3/40** 为 18 胜、0 负、22 平；平均 Teacher 控制约 175 步，15 次窗口因标签失效中止。事件起点例如 motion0 为 142、273、443 帧，显示后续确有多个风险时段。触发器使用 Teacher action 差，属于训练/诊断 oracle，部署不可用。

## 短训设计与结果

两组都从同一个 `model_30098.pt` 续训 100 iterations、256 env，事件规则相同（action L2>1 连续3帧，最多3次、每次80步、冷却40步）。A 组触发后仍由 Student 控制，B 组仅在 Teacher 标签有效时由 Teacher 控制；触发窗口的 PPO gate 在两组均为0，避免把 Teacher 动作当作 Student on-policy PPO。训练期状态分布因此不同，训练 loss 不作同状态性能比较。预检 gate 通过：Teacher 组有 227 次事件、12,651 个有效控制步、19 次标签失效中止。

100 iterations 后均保存 `model_30197.pt`。Teacher 组累计 3,554 次事件、259,136 个有效接管步、259 次失效中止；A 组 3,512 次事件、0 个 Teacher 控制步、1,398 次失效中止。按 100×256×24 计算，B 组 Teacher 实际控制约 42% 训练 transition，是较强干预，不是轻量修补。

无接管 clean full-clip，40 motion×3 episode×4 eval seeds，报告均 complete 且 `intervention.mode=none`：

| seed | 续训前模型 | A：Student 自主续训 | B：恢复片段续训 |
|---:|---:|---:|---:|
| 43 | 10/120 | 0/120 | 2/120 |
| 44 | 6/120 | 0/120 | 9/120 |
| 45 | 14/120 | 0/120 | 5/120 |
| 46 | 9/120 | 0/120 | 4/120 |
| **合计** | **39/480 (8.1%)** | **0/480** | **20/480 (4.2%)** |

B 相对续训前按 seed×motion 为 12 胜、20 负、128 平；失败进度均值 0.531→0.527。**Gate FAIL：保留两组 checkpoint 和所有报告，停止此配置，不继续长训，也不替换较好的源模型。**

## 机制判断与限制

在源 Student 自主 trace 的相同 977 个有效 20%–60% reference 状态上，源→B 的 Teacher action L2 为 **1.4131→1.4126**，几乎不变（21 clip 改善、19 回退）。在 Teacher 辅助轨迹的相同 1,919 个有效状态上为 **0.9626→0.9538**，仅小幅改善。故本轮更像增加了 Teacher 控制状态的覆盖，并未显著提升 Student 已偏离状态上的动作质量；但不能由此唯一证明是观测不足，训练权重、动作目标、PPO 奖励和高接管占比都仍是可检验因素。

下一轮不应直接加长当前配置。建议先审计关键 contact/pull-up 帧的动作幅值、实际力矩和 PPO/DAgger 梯度贡献；再做低占比恢复（例如每回合最多一次、短窗口、保留更多 Student 自主 rollout）的**单变量** A/B，且先用同状态动作误差与 seed43 无接管评估过门槛。若仍回退，转向成功驱动的 PPO 目标或动作/观测表征诊断，而不是继续扩大 Teacher 接管。

代码：`scripts/php_student/{eval_student_full.py,audit_student_release_trace.py,student_recovery_ab.py,run_student_recovery_ab.sh}`。报告与 checkpoint：`reports/php_student/recovery_map_early15_v1/`、`reports/php_student/recovery_ab_v1/`。复核汇总：

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_recovery_ab.sh summary
```
