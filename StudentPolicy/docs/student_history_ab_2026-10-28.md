# Student 时间历史结构 A/B（2026-10-28）

## 问题与假设

当前 Student 在 40 条 motion 的无接管 full-clip 评估中成功率很低；此前检查确认深度分支并非完全失效，Teacher roll-in 改善了 Teacher 状态上的动作拟合，但闭环成功提升有限。本实验只检验一个结构假设：单帧本体观测无法充分辨别障碍过渡阶段的动态状态；给 Student 三帧本体历史是否能改善闭环表现。

## 控制变量

两组都从相同的官方 Student `model_29999.pt` 开始，使用相同 Teacher、深度编码器（单帧）、reward、PPO/DAgger、25%–55% phase Teacher roll-in、256 env、200 次迭代、学习率 1e-4、全新优化器。唯一结构差异是 `actor_obs.history_length`：A=1，B=3。两组均保留原有当前帧权重；B 的额外历史输入权重初始化为零，因此初始策略几乎相同。此实验不是原 checkpoint 优化器状态的精确续训，且两组都作同样处理。

## 已验证

- 原始第一层 140 输入（108 actor + 32 depth）；B 扩展为 356 输入（324 actor + 32 depth）。在随机历史输入上，迁移后第一层与原模型最大差异 `5.72e-06`。
- GPU 预检：A、B 均完成 2 次训练迭代，保存 `model_30000.pt`；这只验证链路可运行，不代表成功率。

## 执行和验收

在 `/home/sjtu/running_man/StudentPolicy` 中：

```bash
export STUDENT_HISTORY_ROOT=/home/sjtu/running_man/StudentPolicy/reports/php_student/history_ab_v1_gpu
bash scripts/php_student/run_student_history_ab.sh verify
bash scripts/php_student/run_student_history_ab.sh preflight
bash scripts/php_student/run_student_history_ab.sh train
bash scripts/php_student/run_student_history_ab.sh eval
bash scripts/php_student/run_student_history_ab.sh summary
```

正式训练输出在 `reports/php_student/history_ab_v1_gpu/train/{single,history3}`。评估固定为 40 条 motion、每条 3 次、seed 43、clean、frame-0、无 Teacher 接管；两组分别写入 `eval/{single,history3}/full_clip_seed43.json`。必须先核对报告 `complete=true` 且 `intervention.mode=none`，再比较总成功数、逐 clip 胜负、失败帧与相位。如果 B 仅降低 Teacher-action error 而不提升无接管 full-clip，则历史输入不足以解释主故障，下一步应检查深度几何/状态可观测性或 DAgger 的数据分布；不能据此宣称结构假设成立。
