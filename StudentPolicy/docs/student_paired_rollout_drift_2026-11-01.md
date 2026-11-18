# Student 续训第 25→50 次的闭环分叉（2026-11-01）

## 实验合同

原模型 `model_30098.pt`、普通续训第 25 次和第 50 次 checkpoint，各在**同一个 Student 评估环境**跑 clean、seed/障碍 seed 43、40 条 low-climb-76 轨迹各 1 回合、frame-0 设置、无 Teacher 接管，并保存每条轨迹的动作、root/anchor、足接触和终止 trace。三组完整报告成功数分别为 **3/40、3/40、0/40**，与此前无诊断 trace 的短评估一致。

采集器记录的首个 motion clock 帧为 **1**，不是 0；40 条轨迹在三组中都从帧 1 连续记录，没有缺帧。两两比较时，40/40 条的首个记录状态中 root 位置、reference 位置、actor observation 和 Teacher observation 全部一致。首帧深度图没有逐帧保存，因此不能宣称所有输入字节完全相同；IsaacSim 的后续仿真也不能当作严格反事实，只能做受控时序诊断。

## 直接对比第 25→50 次

第 25 次成功的是 clip **15、32、38**；第 50 次这三条全部失败。源模型到第 25 次已发生替换：源模型的 clip 3 失去成功，clip 15 获得成功，虽然总数仍为 3/40。因此早期更新已在不同轨迹间产生 trade-off。

| 轨迹 | 首帧动作差 L2 | 首次持续 root 差 ≥2 cm | 首次持续足接触不同 | 首次高位足部支撑：25次→50次 | 末帧：25次→50次 |
|---|---:|---:|---:|---|---:|
| 15 | 0.108 | 25 | 82 | 262 → 未出现 | 868 → 483 |
| 32 | 0.112 | 32 | 54 | 227 → 未出现 | 733 → 508 |
| 38 | 0.113 | 27 | 77 | 280 → 276 | 833 → 489 |

高位支撑按脚踝高于本轨迹起始高度 0.15 m 且同侧足端力 ≥20 N 定义。表中“首次足接触不同”仅是**普通足接触模式**的分叉，不能直接称为障碍接触失败。clip 38 的第 50 次仍出现高位支撑，但之后失败，说明不是所有失败都可归为“未踏上障碍”。全部 40 条的 25→50 配对中，持续 root 差达到 2 cm 的帧中位数约 **29**，持续足接触模式不同的帧中位数约 **72**；这些时间点明显早于上述三条的高位支撑。

结合此前[同状态离线对照](student_fixed_state_checkpoint_drift_2026-11-01.md)：第 50 次 checkpoint 的固定状态动作误差只有小幅普遍变化，而闭环轨迹在早期逐步分叉。**当前最合理的工作假设**是早期接近/步态的微小动作漂移改变了后续接触和障碍进入状态，随后误差累积；不是某个输出突然整体崩坏。先前只把高位支撑前 `[-110,-40]` 帧的 DAgger 权重乘 2，覆盖不到这些从约第 25–32 帧开始的早期状态分叉，这可以解释其没有显著帮助，但不能证明早期分叉是唯一根因。

## 决策

不从第 50 次 checkpoint 继续训练。下一步应优先做**早期轨迹保持**的受控实验，而非再调晚期单帧模仿权重：在相同训练预算内为 frame 0 至障碍前的 actor 动作加源策略行为约束，或限制这一段 PPO 更新，然后检查是否同时保住原成功 clip、没有压制需要学习的新 clip。先过 seed43 逐轨迹 gate，再做 40×3×4 的无接管评估；若只是恢复 3/40 而未超过 39/480 的当前最好完整基线，不能晋级。

产物：[逐轨迹配对摘要](../reports/php_student/continuation_collapse_v1/paired_rollout_v1/paired_drift_direct_v2_summary.json)、[逐帧时间线](../reports/php_student/continuation_collapse_v1/paired_rollout_v1/paired_drift_direct_v2_timeline.csv)；运行入口：[run_student_paired_rollout_drift.sh](../scripts/php_student/run_student_paired_rollout_drift.sh)，分析器：[analyze_student_paired_rollout_drift.py](../scripts/php_student/analyze_student_paired_rollout_drift.py)。
