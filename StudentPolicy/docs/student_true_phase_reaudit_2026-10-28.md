# Student 真正 reference 相位重审计（2026-10-28）

## 修正

旧离线 probe 用失败 trace 的末帧作为相位分母，同时把 `teacher_obs[:, 1:]` 误称为移除路由列；实际路由列为第 449 列。现从评估的 `motion_names.txt` 对应的原始 bundle NPZ 读取完整帧数，以 `frame / (full_frames - 1)` 计算 reference phase，并确实删除 `teacher_obs[449]`。相同修正已用于时序探针的数据加载；旧报告不再作为当前证据。

## 本次实测

- 40 条 Student 自主 trace 在真实 reference phase 40%–60% **没有有效标签样本**；旧的 25%–65% 窗口重算只余 1,713 个样本（旧为 5,539）。所以不能从那些失败 trace 直接判断 40%–60% 阶段的模仿质量。
- 对上述真实相位样本做跨轨迹 5-NN 审计（均衡后 1,688 帧）：actor 输入的跨/同轨迹 Teacher action L2 为 **2.355/1.051**；actor+冻结深度为 **2.459/1.023**；真正去路由列的特权观测为 **1.681/0.964**。这提示跨 motion 的状态/标签差异需要研究，但近邻未严格匹配地形、接触和相位，**不能据此断言不可约观测歧义**。
- 用已有 5 条 Teacher 全程控制、标签有效的 trace，在完全相同的 317 个真实 20%–60% reference 状态上比较两个冻结 Student：原始 `model_29999.pt` action L2 **0.706**，roll-in `model_30098.pt` **0.564**，5/5 motion 均改善。20%–40% 为 0.890→0.669（158 帧），40%–60% 为 0.524→0.461（159 帧）。这证明 roll-in checkpoint 在这些专家状态上确实学得更接近 Teacher，**但不能推论其自主闭环会稳定成功**；后者四个评估 seed 仅 25/480。
- 在 40 条 Student 自主 trace 的真实 20%–40% 状态上，原始→roll-in 的配对 action L2 为 **1.372→1.351**（3,351 帧），改善很小；40%–60% 无样本。说明专家状态上明显降低动作误差，与 Student 自身偏离状态上几乎没有降低，是当前最值得验证的分布差距。
- 另对 roll-in checkpoint 在同一个 seed43 做 40 条 motion × 首回合、无接管的 paired-action trace。两模型的首回合都为 **0/40**；既有 3 回合评估中的 roll-in 6/120 成功不能误写为“首回合成功”。以 action L2 > 1.0 连续 10 帧的旧阈值衡量，首次持续分歧帧中位数原始 **211.5**、roll-in **188.0**；逐 motion 有 31 条变早、5 条变晚（4 条相同）。tracking 失败帧中位数 436.5→432.5。该分歧阈值是描述性指标，不能单独证明因果，但与“只在专家状态上拟合改善、自主闭环未改善”一致。

## 下一项实验 gate

**后续已执行：**15%–55% 提前连续 roll-in 的同源短 A/B 与四个 eval seed 结果见 [student_early_segment_rollin_ab_2026-10-29.md](student_early_segment_rollin_ab_2026-10-29.md)。本节以下为当时的实验建议，不能将它误读成尚未实施。

保持两个 checkpoint 不变，针对相同 motion 的 Teacher 控制状态与 Student 自主状态继续按真实 reference 帧、接触模式和地形分层，确认早期分歧的机制。现有证据支持优先试连续片段级恢复数据，而非继续增加同类专家状态监督；短训后必须通过无接管多 seed full-clip gate。当前结果不支持直接长训或修改成功阈值。

数据：`reports/php_student/own_state_supervision_probe_v1/{label_conflicts_true_phase_no_routing_v2,fixed_teacher_states_source_vs_rollin_true_phase,fixed_state_source_vs_rollin_true_phase}.json`。脚本：`scripts/php_student/{probe_observation_vs_state,audit_student_label_conflicts,probe_student_temporal_observability,compare_fixed_student_states}.py`。

首回合闭环 trace 与分歧审计：`reports/php_student/rollin_true_phase_trace_seed43/{full_clip_seed43,first_divergence}.json`、`reports/php_student/own_state_supervision_probe_v1/source_first_divergence_true_phase.json`。
