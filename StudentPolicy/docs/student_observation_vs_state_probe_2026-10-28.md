# Student 观测与状态信息判别实验（2026-10-28）

## 问题与控制变量

问：Student 难以独立越障，是否因为其深度观测/表征无法预测同状态下 Teacher 的动作？在同一批 Teacher 控制状态上读取同帧 `actor_obs`、Student 实际收到的深度图、`teacher_obs` 与 Teacher 动作；目标动作和样本完全相同，只改变回归器的输入。留出**整条 motion**，避免相邻帧随机切分造成泄漏；另报同轨迹连续时间块留出。所有离线模型仅用于诊断，不部署。

评估数据：seed43、clean、40 条 motion × 1 回合、Teacher 全程接管，阳性对照成功 38/40；保存 40 条 paired trace。困难相位 25%–65% 对齐到 2,542 个 Teacher 标签有效、深度图与本体状态同帧的样本。四组输入：本体观测；本体观测加 32 维全零（匹配输入宽度）；本体观测加原 Student checkpoint 的**冻结**深度编码特征；privileged `teacher_obs`。各组使用同结构的两层 128 宽回归器、相同训练步数和按训练集计算的标准化。种子 42、43 均跑完 40 折整轨迹留出。

## 结果

动作 L2，越低越好：

| 输入 | seed42 留出均值 | seed43 留出均值 |
|---|---:|---:|
| actor only | 0.483 | 0.475 |
| actor + zero latent | 0.479 | 0.477 |
| actor + frozen depth latent | 0.510 | 0.502 |
| privileged teacher_obs（新训小模型） | 0.565 | 0.563 |

两个种子、40 条留出轨迹中，真实冻结深度特征相对同维度零特征均为 **0/40 胜**。原 checkpoint Student 在这些 Teacher 控制状态上的动作 L2 约 0.680。5 条轨迹的先导试验与同轨迹连续块留出也没有看到冻结深度特征收益。

## 严谨解释

当前证据**不支持**“只要增加现有深度 latent，Teacher 动作就更可预测”；它更支持排查深度编码器、特征融合及训练利用方式。它**不能证明**原始深度图无信息：此处 backbone 被冻结，且仅有单帧稀疏取样。privileged 小模型留出误差较高也不能证明状态信息没用：Teacher 动作本身由已训练 Teacher 根据 privileged 状态生成，而新训小模型只有有限样本和 600 步，跨 motion 泛化并非等价任务。

下一步应在相同样本与整轨迹留出下比较：冻结深度 backbone、可训练深度 backbone，以及仅本体观测；同时报告训练/留出误差与在线无接管成功率。只有离线特征收益能稳定转化为在线成功，才考虑把它作为 Student 主干修改。

代码：`scripts/php_student/probe_observation_vs_state.py`。报告：`reports/php_student/observation_state_probe_seed43_all40_v2.json`、`v3.json`；paired trace：`reports/php_student/teacher_state_action_audit_seed43_all40/traces/`。

## 可训练深度 backbone 后续验证

上述下一步的离线部分已执行。用同样的 2,542 个样本，按 motion 分组 5 折交叉验证，三组使用同一 128×128 动作回归头与相同训练预算；可训练组从原 Student checkpoint 的深度 backbone 初始化，head 学习率 `3e-4`，backbone 学习率 `3e-5`，每折 600 更新。种子 42/43 同时改变整轨迹分组和初始化。

| 输入 | seed42 留出动作 L2 | seed43 留出动作 L2 |
|---|---:|---:|
| actor + 32 维零特征 | **0.481** | **0.482** |
| actor + 冻结深度 backbone | 0.506 | 0.508 |
| actor + 可训练深度 backbone | 0.491 | 0.497 |

可训练深度确实改善了冻结深度，但两个种子均未超越同维度零特征。**因此本轮没有通过“视觉特征相对无视觉对照产生留出收益”的进入在线训练门槛；未启动新的 Student 续训。** 这不能证明深度图无信息，只说明在当前编码器初始化、样本预算与纯动作回归目标下，视觉未转化为更好的跨 motion 预测。下一项更有信息量的检查是：在相同数据上测深度是否能预测障碍几何/接触时刻，并检查当前 Student 实际深度 latent 的变化是否与这些事件对齐；若几何可预测而动作收益仍无，优先改融合/蒸馏目标，而非盲目加训练轮数。

代码：`scripts/php_student/probe_trainable_depth.py`。完整结果：`reports/php_student/trainable_depth_probe_seed42_v1.json`、`trainable_depth_probe_seed43_v1.json`。
