# Student 固定状态回放 A/B（2026-11-02）

## 目的与数据门槛

前缀动作锚定能减小同状态漂移却不能保住成功率；早期 PPO×0.1 也无效，而普通训练末期参考帧 `<100` 在 rollout 中只占约 3%。本轮从最佳 Student `model_30098.pt` 出发，检验固定比例状态回放是否能分别改善**早期能力保持**与**过渡阶段 Teacher 监督**。两种干预分开训练；共同对照为先前的 `gate_rebuild_ab_v1/control_discard`（同源 checkpoint、256 env、训练 seed 42、50 次迭代，同一 gate 状态加载后清空）。各组保存第 25/50 次 checkpoint。

数据来自最佳 Student 自主 clean rollout 的 40 条逐轨迹 trace；只使用精确对齐到深度图的帧，按 clip 均衡抽样，每次 optimizer step 固定抽 256 个回放状态。前缀组使用参考帧 0–99 的 **760** 个状态，目标为冻结的原 Student 动作，MSE 系数 50；过渡组使用参考帧 100–249 且 Teacher 标签有效的 **1,147** 个状态，目标为 Teacher 动作，MSE 系数 10。40/40 条轨迹均有样本，过渡区有效标签率 95.6%；原 checkpoint 对记录动作的最大重放误差低于 `6e-7`。过渡区原 Student–Teacher 平均动作 L2 约 0.999。预检确认回放损失非零、checkpoint 正常生成。

## 结果

| 模型 | seed43 40×1 clean 无接管 | seed43 40×3 clean 无接管 | 固定状态机制指标 |
|---|---:|---:|---|
| 原最佳模型 | 3/40 | **10/120** | 对照基线 |
| 普通续训对照 50 次 | 2/40 | 7/120 | 早期相对原模型动作漂移 L2 0.158；过渡 Teacher 动作 L2 1.011 |
| 固定前缀回放 25 次 | 7/40 | 14/120 | 尚未做离线漂移审计 |
| 固定前缀回放 50 次 | 5/40 | **16/120** | 早期动作漂移 L2 **0.047**（比对照低约 71%） |
| 过渡 Teacher 回放 50 次 | **0/40** | 未扩展 | 过渡 Teacher 动作 L2 **0.821**（比对照低约 19%） |

前缀回放的 50 次候选虽在这个 seed 上比原模型多 6/120、比对照多 9/120，但原模型 clip 32、38 均为 **3/3 → 0/3**；第 25 次也让 clip 32 **3/3 → 0/3**、clip 3 **1/3 → 0/3**。它是有局部正效应和明显遗忘并存的研究候选，**不是合格的新基线**。过渡 Teacher 回放在固定旧状态上确实更像 Teacher，但自主闭环 0/40，按 gate 停止。所有报告为 `complete=true`；40×1 仅用于初筛，40×3 用于同 seed 复核，尚无多 seed 正式成绩。

## 解释、决策与后续

1. 固定比例回放比只依赖当次 rollout 更能保持早期动作，并在 seed43 提高部分轨迹成功率，说明**训练覆盖**是值得继续研究的因素；但增益以丢失原有强项为代价，不能仅用总数宣布提升。
2. 对旧的 Student 过渡状态做单帧 Teacher MSE，降低了动作误差却没有带来闭环恢复。有效标签率高只说明这些位置有监督，不保证它覆盖实际新策略将进入的偏差状态，也不保证单帧模仿能形成接触时序协调。
3. **STOP：**不延长这两组训练、不组合两个损失碰运气、不用这些 checkpoint 取代 `model_30098.pt`。下一轮应设计能同时保留原成功 clip 和改善失败 clip 的闭环分布更新：按轨迹/接触阶段平衡的在线状态采集与短时序恢复目标，并预先锁定逐 clip 无回退 gate。只有先在多个 seed 下保住原强项、再超过原正式 **39/480**，才晋级。

代码：[训练与回放实现](../scripts/php_student/train_student_phase_ab.py)、[启动封装](../scripts/php_student/run_student_phase_ab.sh)、[固定状态动作审计](../scripts/php_student/audit_student_early_anchor.py)。报告：[前缀 50 次 3 回合](../reports/php_student/fixed_replay_ab_v1/eval/prefix_seed43_3ep/full_clip_seed43.json)、[前缀 25 次 3 回合](../reports/php_student/fixed_replay_ab_v1/eval/prefix_step025_seed43_3ep/full_clip_seed43.json)、[对照 3 回合](../reports/php_student/fixed_replay_ab_v1/eval/control_seed43_3ep/full_clip_seed43.json)、[过渡 1 回合](../reports/php_student/fixed_replay_ab_v1/eval/transition_seed43/full_clip_seed43.json)、[前缀动作漂移](../reports/php_student/fixed_replay_ab_v1/prefix_action_drift_vs_control.json)。
