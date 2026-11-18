# Student 预接触 DAgger 2× A/B（2026-11-01）

## 问题与实验

此前的三次 rollout 复核显示：障碍预接触窗口有大量有效 Teacher 标签，但 Student 动作误差仍高；PPO actor 梯度远强于 DAgger 梯度。为了检验“只增强该窗口的模仿信号”是否有效，保持官方 PHP/Holosoma Student trainer、源 checkpoint、训练 seed 42、256 env、100 次继续训练及其余配置不变，做两组：

| 组 | 改动 |
|---|---|
| source | 不继续训练，直接评估 `model_30098.pt` |
| baseline | 继续训练；预接触有效 Teacher 标签的 DAgger 权重 1× |
| focus | 同样继续训练；唯一计划变量是上述样本的 DAgger 权重 2× |

预接触窗口定义为 Teacher 轨迹首次“踝部高于起始高度 0.15 m 且足端接触力 ≥20 N”的帧之前 `[−110,−40)` 帧；Teacher 全零动作（lost expert）仍被掩码。40 条源轨迹及 3 组变体的训练顺序在运行时校验通过。40 条 Teacher trace 中 38 条检测到该事件；理论窗口共 2,660/31,640 参考帧（8.4%）。这**不是**实际训练采样占比，当前日志没有记录命中数，不能据此声称 2× 实际覆盖了多少梯度样本。

## 评估与结果

先做 seed 43、每条轨迹 1 次的 clean frame-0 无接管短评估：source 3/40，baseline 0/40，focus 0/40。按预定停训规则保存两组 checkpoint，再做完整评估：40 条轨迹 × 每条 3 次 × seed 43–46；障碍 seed 与评估 seed 对应，所有报告完整，Teacher takeover 为 `none`。

| 组 | seed 43/44/45/46 成功数（各 120 次） | 合计 | 门槛 |
|---|---|---:|---|
| source | 10 / 6 / 14 / 9 | **39/480 = 8.1%** | 对照 |
| baseline | 0 / 0 / 0 / 0 | **0/480** | 回退 |
| focus | 0 / 0 / 0 / 0 | **0/480** | 未胜过 baseline，回退 |

判定：**STOP，不延长训练，不采用 focus checkpoint。** 训练内平均回报在两组均上升，但 clean full-clip 闭环成功率归零；训练指标不能替代验收。

这只能说明“当前 100 次继续训练 + 局部 2× DAgger”无效；不能断言 DAgger 加权本身有害，因为 baseline 也同样回退。更应先检查继续训练时的有效梯度/采样分布、优化器与学习率恢复、PPO gate、depth backbone 更新和训练/clean 评估分布差异，再考虑新的权重 A/B。下一次 A/B 必须记录窗口命中数、有效 Teacher 标签数、窗口内 action error 和 PPO/DAgger 梯度贡献，并先以 10–25 次迭代作早停 checkpoint sweep。

## 复现位置

- 训练入口：[run_student_phase_ab.sh](../scripts/php_student/run_student_phase_ab.sh)；权重实现：[train_student_phase_ab.py](../scripts/php_student/train_student_phase_ab.py)。
- 评估入口：[run_precontact_dagger_ab_eval.sh](../scripts/php_student/run_precontact_dagger_ab_eval.sh)；汇总校验：[summarize_precontact_dagger_ab.py](../scripts/php_student/summarize_precontact_dagger_ab.py)。
- 结果：[full_eval_summary.json](../reports/php_student/precontact_dagger_ab_v1/full_eval_summary.json)，逐 seed JSON 在同目录的 `full_eval/{source,baseline,focus}/seed*/`；A/B checkpoint 分别在 `baseline/model_30197.pt` 与 `focus/model_30197.pt`，源 checkpoint 保持不变。
