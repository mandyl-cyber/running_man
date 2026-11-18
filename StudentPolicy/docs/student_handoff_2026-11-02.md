# Student 蒸馏研究交接（2026-11-02）

## 一页结论

目前能完成 low-climb 任务的是 **Teacher**，不是部署形态的深度 **Student**。官方 PHP/Holosoma Teacher 在 40 条 motion、每条 100 次的 clean frame-0 full-clip 评估中为 **3837/4000（95.9%）**；当前已测最佳 Student `model_30098.pt` 在 40 条×3 次×4 个 seed（43–46）、clean、**无 Teacher 接管**下为 **39/480（8.1%）**。两者是不同观测与控制条件下的策略，不应把 Teacher 或接管的成功率写成 Student 成果。

最可信的问题链是：**Student 早期动作/状态偏差 → 接触过渡被放大 → 当前 DAgger 的 Teacher 路由把大偏差状态标为失效并屏蔽标签 → 后续恢复监督变少**。同时，Student 的可用观测与 Teacher 特权观测之间存在离线动作拟合差距。哪一个是首要根因尚未由受控训练 A/B 确定；不能宣称“只缺更多 Teacher”或“只缺 DAgger”。

## 工作位置与不可混淆的产物

- 工作仓库：`/home/sjtu/running_man/StudentPolicy`；训练主体依赖 `/home/sjtu/php_parkour_official` 的 PHP/Holosoma，`scripts/php_student/` 是本地实验与评估封装。数据为 `/home/sjtu/.cache/php-parkour/motions-low-climb-76` 的 40 组 motion–terrain 配对文件。
- Teacher：`/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261015_072805-php-teacher-seed42-locomotion/model_19999.pt`；[40-clip clean 报告](../../new_teacherpolicy/reports/php_official/teacher_full_eval_clean/full_clip_seed42.json)。
- 当前最佳 Student：`reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt`；正式评估在 `reports/php_student/teacher_rollin_early15_ab_v1/eval/teacher/`。它只是**当前相对最好**，远未验收通过。
- 本轮三个结构测试：[实验说明](student_structural_tests_2026-11-02.md)、[汇总 JSON](../reports/php_student/structural_tests_v1/summary.json)、[统一脚本](../scripts/php_student/run_student_structural_tests.sh)。测试中生成的 `single_clip/train/model_final.pt` **不晋级**。

## 已做实验与真正得到的结论

| 阶段/干预 | 可信结果 | 不能据此得出的结论 |
|---|---|---|
| 官方式 Student 训练 | 原 `model_29999.pt` 的已测 4-seed 自主评估为 **0/480**；Teacher 同技能 clean 很强。[roll-in 记录](student_teacher_rollin_ab_2026-10-28.md) | Teacher 强不代表 Student 蒸馏成功。 |
| Teacher roll-in／提前覆盖接近段 | 25%–55% 窗口的候选约 **25/480**；提前到 15%–55% 的当前最佳为 **39/480**。[提前 roll-in](student_early_segment_rollin_ab_2026-10-29.md) | 只是弱正效应，且尚未形成鲁棒的自主策略。 |
| 有限/多次 Teacher 接管 | 早期恢复测试有明显辅助增益；本轮同 seed 40 条中 **无接管 3/40、提前接管 80 步后 13/40**。[结构测试](student_structural_tests_2026-11-02.md) | 13/40 是 Teacher 辅助，不是 Student 自主成功；当前 Teacher 也救不回所有状态。 |
| 预接触 DAgger×2、Teacher 控制恢复片段续训 | 预接触加权两组正式评估均 **0/480**；恢复续训候选 **20/480**，低于源模型 39/480。[DAgger A/B](student_precontact_dagger_ab_2026-11-01.md)、[旧交接](student_handoff_2026-10-29.md) | 不能说 DAgger 本身有害；普通对照续训也回退。 |
| 续训回退与 PPO gate | 零更新加载重存不回退；训练约 26–50 次后出现回退。checkpoint **没有保存相位 gate 的环境统计状态**；重建近似 gate 与丢弃 gate 的短评估同为 **2/40**。[回退定位](student_continuation_collapse_probe_2026-11-01.md)、[gate A/B](student_ppo_gate_rebuild_ab_2026-11-02.md) | gate 丢失是真缺口，但不是已证实的唯一主因；不能继续用旧续训结果做单变量归因。 |
| 早期 PPO 降权、动作锚定、固定状态回放 | PPO×0.1 为 **2/40**；锚定把前缀同状态漂移降约 65%，闭环仍 **0/40**。前缀回放达到 seed43 **16/120**，却使原成功 clip 32/38 从 **3/3→0/3**；过渡 Teacher 回放降低动作误差但闭环 **0/40**。[PPO A/B](student_early_ppo_gate_ab_2026-11-02.md)、[锚定](student_early_action_anchor_ab_2026-11-01.md)、[回放](student_fixed_replay_ab_2026-11-02.md) | 固定状态动作误差或总成功数的小幅增加，不足以允许逐 clip 遗忘。 |
| 2026-11-02 单轨迹/观测/恢复结构测试 | 同一配对 clip、seed43：Teacher **20/20**，Student 短续训前后 **0/20**；Teacher 特权观测离线预测误差 **1.08**，Student 输入约 **1.94**；250 帧后有效标签仅 **2641/8163（32.4%）**。[完整记录](student_structural_tests_2026-11-02.md) | 单轨迹只在 `1e-5` 学习率下续训 200 次，不能排除更充分单轨迹训练会成功；离线误差不能证明原始深度图信息不足。 |

## 当前训练代码实际如何使用 DAgger

**已使用 DAgger + PPO，不是只用 PPO。** 最佳 checkpoint 保存的配置是 `dagger_loss_coef=10`、`distillation_warmup_steps=0`、`ppo_start_epoch=0`、`dagger_end_epoch=10000`；迭代 30098 时 `ppo_coef=0.9`、学习率 `1e-5`，Teacher 权重已加载。有效样本上的 DAgger 系数是 `(1−gate×ppo_coef)×dagger_loss_coef`，但梯度贡献不能仅凭系数大小判断。详见 [保存配置](../reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/holosoma_config.yaml) 和官方 `holosoma/agents/distillation_ppo/distillation_ppo.py`。

关键机制在官方 `wbt_training/config_values/motion_command.py::which_motion`：当前机器人与 reference 的 anchor/body/orientation 误差超阈值时，路由编号变为 `-1`。`wbt_training/utils/student_teacher.py` 随后将 Teacher 动作置零，DAgger 对该行损失置零，**批次分母仍包含该行**。这个标记是跟踪误差启发式，**不是**实际测试 Teacher 能否从该状态救回的判定。因此不能直接放开掩码，也不能把零动作当成正确恢复标签。

## 下一步：先验证恢复监督，再做受控训练

### G0：冻结策略，建立可恢复性数据集（不改 Student）

在当前最佳 Student 的 clean、自主 rollout 中，按每条 motion 与多个 seed 保存首次持续动作分歧附近的完整仿真状态、正确 clip ID、reference 帧、配对 terrain、Student 输入与实际动作。至少覆盖分歧前、分歧时和首次 `which_motion=-1` 的状态。**把 Teacher 选择 ID 与标签有效性分开**：已知 clip ID 用于正确调用 Teacher；原来的失效标记保留作诊断，不直接决定把动作写入训练集。对每个保存状态做 Teacher 短接管并交还 Student 的分支测试，只把能重新跟踪且完成 full clip 的分支标为“经验证可恢复”。同时报告不可恢复比例、接管时间、动作误差、各阶段和各 clip 的样本量。此 Gate 先回答：失效标签里有多少其实是可救状态。

### G1：如现有 Teacher 可恢复性不足，先强化 Teacher

本轮 80 步接管仅挽回 10/40 条，不能假定它能给所有失效状态提供正确监督。若 G0 的已验证恢复分支不足，应从 Student 偏离状态重置训练**同一技能的恢复 Teacher**，保持 motion–terrain 绑定和正确相位；用独立状态与 seed 复测恢复成功率。此处不是先为每条 clip 增一个 Teacher，也不是将所有 `-1` 状态强制标为有效。

### G2：恢复数据 DAgger 的严格 A/B

从**同一较早且有学习率余量的 Student checkpoint**出发；若只能用 `model_30098.pt`，两组必须同样重设学习率并处理 PPO gate 状态。控制组按原 DAgger+PPO 训练；实验组仅额外加入 G0/G1 通过的恢复状态与 Teacher 动作，按 clip/阶段均衡采样。两组保持相同 env、种子、更新量、PPO 与原训练数据，记录有效标签数、恢复样本占比、DAgger/PPO 梯度及前 50 帧动作/状态漂移。先做少量更新与早停 checkpoint sweep；若固定状态误差下降却闭环无提升或原成功 clip 回退，立即停止，不延长训练。

### G3：只用无接管闭环结果晋级

先在 seed43、40×1 上筛查，再对候选做与基线完全相同的 **40 clips×3 episodes×seeds43–46** clean、frame-0、`intervention.mode=none` 评估，并检查 `complete=true`。候选须超过 **39/480**，且不能以原有成功 clip 大幅回退换取总数增长；报告逐 clip 胜负、失败帧、标签有效率及 seed 方差。辅助接管成功率只作机制指标。通过后再扩随机化鲁棒性、跨技能和 sim-to-sim；当前没有 Student checkpoint 通过这条验收链。

## 复现与交接注意

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_structural_tests.sh preflight
bash scripts/php_student/run_student_structural_tests.sh summary
```

完整重跑三个结构测试可用 `bash scripts/php_student/run_student_structural_tests.sh all`，但训练/仿真需 CUDA/Isaac Sim，且会耗时；新条件须指定新的 `STUDENT_STRUCTURAL_ROOT`，脚本会拒绝混用不同输入。当前工作树有大量未跟踪的既有实验文档与脚本，接手前检查 `git status`，不要误删、混入无关改动或把辅助成功的 checkpoint 当成正式 Student 基线。
