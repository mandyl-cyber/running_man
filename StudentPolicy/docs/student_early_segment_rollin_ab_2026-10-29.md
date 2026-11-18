# Student 提前连续 roll-in 短实验（2026-10-29）

## 因果问题

原始 Student 的 40 条首回合 trace 中，15 条在真实 reference phase 25% 前已出现持续动作分歧。旧 roll-in 从 25% 才接管，可能开始过晚。本实验将窗口提前为 **15%–55%**，窗口内 Teacher 接管概率固定为 1.0、仅在标签有效时接管。因而是连续的 phase 片段，并非逐步随机掷币；失效标签状态仍不会被强行标注为有效。它**不是**从已失效状态恢复的实验。

两组均从原始 Student `model_29999.pt` 续训 100 iterations、256 env、相同 PPO/DAgger 配置和 15%–55% gate；A 组全程 Student 控制，B 组在窗口内 Teacher 控制。原始 checkpoint 和两个新 checkpoint 均保留。

## 机制预检与训练

- 预检 15%–25%：B 组 Teacher 实际接管 7,146 个有效状态；40%–60% 有效标签率 A **55.7%**、B **97.7%**，机制 gate 通过。
- 训练累计 15%–25% 接管 35,362 个有效状态；20%–40% 标签有效率 **54.4%→96.9%**，40%–60% **69.9%→98.2%**。
- 训练期有效标签上的 action L2，20%–40% 为 **1.364→0.698**，40%–60% 为 **1.041→0.593**。这些是不同闭环状态分布上的统计量，不可直接当作“同状态动作误差下降”。

## 无接管 full-clip 评估

clean、40 motion × 每条 3 episode、评估 seed 43–46；报告均 `complete=true` 且 `intervention.mode=none`。旧 25%–55% 模型是此前同源、同预算实验；seed 43/44 另评估了本次 Student 控制对照组。

| seed | 本次 Student 对照 | 旧 25%–55% roll-in | 新 15%–55% roll-in |
|---:|---:|---:|---:|
| 43 | 0/120 | 6/120 | 10/120 |
| 44 | 0/120 | 6/120 | 6/120 |
| 45 | 未测 | 4/120 | 14/120 |
| 46 | 未测 | 9/120 | 9/120 |
| 合计 | 0/240 | **25/480 (5.2%)** | **39/480 (8.1%)** |

旧→新逐 seed×motion 的成功回合数比较为 **21 胜 / 16 负 / 123 平**；失败回合的每 motion 平均进度约 **0.523→0.531**。增加的 14 次成功主要来自 seed43 与 seed45，seed44/46 没有总数提升。只做了一个训练 seed；评估回合并非独立训练重复。

**Gate 结论：机制 PASS，部署性能 FAIL，候选 HOLD。** 提前接管产生弱正效应，但 8.1% 成功率及轨迹级回退不足以晋升为可用 Student，不继续同配置长训。下一步应检验训练 seed 重复性，或设计真正包含 Student 偏离状态的片段级恢复数据；必须继续以无接管多 seed full-clip 为最终门槛。

结果目录：`reports/php_student/teacher_rollin_early15_ab_v1/`；旧窗口对照：`reports/php_student/teacher_rollin_ab_v3/`。代码：`scripts/php_student/teacher_rollin_ab.py`（p=1.0 显式确定接管，并记录 15%–25% 计数）；现有 `run_teacher_rollin_ab.sh` 通过环境变量设置 fresh root 与窗口。
