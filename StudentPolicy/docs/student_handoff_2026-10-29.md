# Student 蒸馏实验交接（2026-10-29）

> **2026-11-02 最新状态（优先于下文较早的待办）：**最佳 Student 仍是 `model_30098.pt`，clean、40 clips×3×4 seeds、frame-0、无接管为 **39/480**。预接触 DAgger×2、前 100 帧源策略动作锚定、重建近似逐相位 PPO gate，以及前 100 帧 PPO×0.1 均已做短训 A/B，**没有候选通过闭环晋级门槛**；不要继续这些 checkpoint 的长训。gate 状态未随 checkpoint 保存是实在的流程缺口，但重建后与丢弃对照同为 seed43 2/40；早期 PPO 降权也为 2/40，前缀同状态漂移只降约 2.7%。源策略动作锚定虽然使前缀漂移降约 65%，闭环仍为 0/40。下一步应先处理早期状态训练覆盖不足（训练末期前缀占比约 3%）及障碍过渡恢复监督；不要把固定状态 MSE 或 Teacher 接管成功算作自主进步。详见 [gate A/B](student_ppo_gate_rebuild_ab_2026-11-02.md)、[早期 PPO A/B](student_early_ppo_gate_ab_2026-11-02.md)、[动作锚定 A/B](student_early_action_anchor_ab_2026-11-01.md)。

> **同日后续：**已做固定比例状态回放的两个独立短训。前缀原 Student 动作回放让 seed43 的 3 回合结果从原模型 **10/120**、普通续训 **7/120** 到 **16/120**，但原模型 clip 32/38 均 **3/3→0/3**；第 25 次早停仍有 clip 32 严重回退。过渡阶段有效 Teacher 动作回放把目标状态上的动作 L2 从 **1.011→0.821**，闭环却 **0/40**。两组均未达到无明显逐 clip 回退的晋级门槛；最佳模型不变，未做多 seed 正式评估。具体数据与下一步边界见 [固定状态回放 A/B](student_fixed_replay_ab_2026-11-02.md)。

## 一句话现状

Teacher 的低墙攀爬能力已可作为蒸馏来源，但当前深度 Student **尚未达到可用的自主闭环成功率**。本地最好短实验 checkpoint 在 clean、40 条 low-climb-76 motion、frame-0、无 Teacher 接管、评估 seeds 43–46、每条每 seed 3 回合下为 **39/480（8.1%）**。最新“多事件恢复片段”短训只得 **20/480（4.2%）**，已判定回退并停止继续训练；不得把它替换为当前最好模型。

**2026-10-30 更新：**关键窗口力矩及全关节短时接管、storage gate 与真值相位对照均已复核。全关节 Teacher 接管可救回部分失败，但它不是 Student 能力；真值相位在 40 条整轨迹留出、两种子离线动作预测中无收益。当前最合理的工作假设是接触过渡阶段的观测/全身动作/闭环恢复耦合失配，不能唯一归因。已执行结果、限制和下一个分阶段 Gate 见 `docs/student_root_cause_plan_2026-10-30.md`；不要直接启动新一轮长训。

同日又完成足部接触事件可观测性先导：对齐深度比同轨迹打乱深度略好，但未超过本体观测基线；现有 trace 无手接触标签，尚不足以判断攀爬关键事件的视觉可观测性。详见上述 Gate 文档，仍不要据此修改线上视觉结构。

随后补采全身接触力与脚踝位置：38 条 Teacher 成功轨迹中 23 条无 >10 N 腕部代理接触；低墙任务以足/腿支撑为主。首次高位足部支撑的离线预测由本体观测即可达到 AUC≈0.968，简单拼接深度未改善。特别要注意 Student 在已分析的 motion 4/12/24 上先于 Teacher 的高位支撑事件发生持续动作分歧，因此下一道训练信号审计应移到**预接触准备窗口**，而非只加权落脚后的帧。此结果仍不是 Student 自主性能增益。

**G2 更新：**512 env×70 rollout 的无更新复采已验证训练映射为 40 源 motion×3 terrain variants，采集了 860,160 transition，PPO gate 已启动。预接触代理窗有效标签率 93.4%、动作 L2 1.198、有效 DAgger 权重 1.631；早期对照动作 L2 0.748、DAgger 权重 5.374。三次 rollout 中 PPO actor 梯度范数远高于 DAgger，方向近正交且轻微负相关；这是当前 checkpoint 的新 rollout 瞬时梯度，不代表历史累计梯度。已决定下一步只做预接触窗 DAgger×2 的短 A/B 预检与 100-iteration 对照，**尚未启动训练**。细节及停止 Gate 见 `docs/student_root_cause_plan_2026-10-30.md`。

## 当前基线与执行位置

- 工作仓库：`/home/sjtu/running_man/StudentPolicy`。实验读取 `/home/sjtu/php_parkour_official` 的 PHP/Holosoma 实现和缓存的 `motions-low-climb-76` motion/terrain bundle；本地 `scripts/php_student/` 是诊断与受控实验封装，**不等于未经修改的官方训练流程**。
- 当前最好（仅相对本项目已测候选，非部署合格）：`reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt`。评估：`reports/php_student/teacher_rollin_early15_ab_v1/eval/teacher/full_clip_seed{43,44,45,46}.json`。
- 原始官方式 Student checkpoint：`/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt`。它的自主 full-clip 表现远低于 Teacher，不应因 Teacher 具备完成能力而推定 Student 已学会。
- 最新 A/B 两个 checkpoint：`reports/php_student/recovery_ab_v1/train/{student,teacher}/model_30197.pt`；评估：`reports/php_student/recovery_ab_v1/eval/{student,teacher}/seed{43,44,45,46}/full_clip_seed*.json`。这些 checkpoint **保留作失败分析，不晋级**。

## 到目前为止可信的证据

1. 先前脚本的 reference phase 曾用失败 trace 末帧作分母，并错删 Teacher 观测路由列；已按原 bundle 总帧数和实际路由列修正。旧相位统计不能沿用。详见 `docs/student_true_phase_reaudit_2026-10-28.md`。
2. 提前连续 roll-in（15%–55% phase）相对旧窗口（25%–55%）将无接管 full-clip 从 **25/480→39/480**，但只有一个训练 seed，seed44/46 没有总数增益；这是弱正效应，不是 Student 验收通过。详见 `docs/student_early_segment_rollin_ab_2026-10-29.md`。
3. 诊断性 Teacher 接管能救很多轨迹，但它替换了 Student 的动作：较新 checkpoint 无接管 seed43 为 **3/40**；首次分歧点接管80步并交还为 **14/40**；最多3次×80步、冷却40步的辅助完成 **21/40**，平均约175个 Teacher 控制步。触发器依赖 Teacher action 差，不能用于 Student 部署。详见 `docs/student_recovery_feasibility_map_2026-10-29.md` 和 `docs/student_recovery_ab_2026-10-29.md`。
4. 最新受控短训从同一 `model_30098.pt` 开始、各100 iterations／256 env，A 由 Student 控制、B 在有效标签恢复窗口内由 Teacher 控制；两组触发规则与 PPO gate 相同。预检通过，B 组累计约259,136个 Teacher 控制步（约42%训练 transition），但自主评估如下：

   | seed | 源模型 | A 自主续训 | B 恢复续训 |
   |---:|---:|---:|---:|
   | 43 | 10/120 | 0/120 | 2/120 |
   | 44 | 6/120 | 0/120 | 9/120 |
   | 45 | 14/120 | 0/120 | 5/120 |
   | 46 | 9/120 | 0/120 | 4/120 |
   | 合计 | **39/480** | **0/480** | **20/480** |

   所有这张表中的评估均为 `complete=true`、`intervention.mode=none`。B 相对源模型为12胜／20负／128平（seed×motion），失败进度均值0.531→0.527；**Gate FAIL**。
5. 在相同的977个源 Student 自主状态上，源→B 的 Teacher action L2 **1.4131→1.4126**，几乎没改善；在1,919个 Teacher 辅助状态上 **0.9626→0.9538**，改善很小。说明“扩大有效标签覆盖”并未自动转化成 Student 在自身偏离状态上的动作改进；但不能据此单独断言观测不可辨或某一个超参数是唯一根因。

## 接下来先做什么

**不要继续最新 B 组长训，也不要再用接管成功率做 Student 验收。** 下一步优先做一个离线／只读的关键帧审计：按真实 reference 帧和接触事件对齐 Teacher 与最好 Student 的动作幅值、实际关节力矩、pelvis 高度/速度、手足接触、深度观测和 PPO/DAgger 权重，特别检查 pull-up 的短促高力矩动作是否被 Student 低估。同时检查新训练的 Teacher 控制占比42%及零 PPO gate 是否让成功驱动信号过少。这里是**待验证假设**，不是现已证明的原因。

若审计指向接管占比过高，才从同一最好 checkpoint 做更低占比、单变量短 A/B（例如每回合最多一次、较短窗口）；先查同状态动作误差与 seed43 无接管结果，不改善就停止。若指向关键力矩动作低估，则优先对照 PHP 的 DAgger+PPO 成功驱动目标，而不是再堆 Teacher 正常轨迹。任何候选最终须回到相同的40×3×4无接管 full-clip评估；至少超过当前39/480、跨 seed 不集中回退，才能考虑延长训练。当前没有候选通过该门槛。

## 复核入口与注意事项

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_recovery_ab.sh summary
```

完整实验与机制说明：`docs/student_recovery_ab_2026-10-29.md`；恢复能力地图：`docs/student_recovery_feasibility_map_2026-10-29.md`。主要脚本：`scripts/php_student/{eval_student_full.py,eval_student_full.sh,run_student_recovery_map.sh,student_recovery_ab.py,run_student_recovery_ab.sh,compare_fixed_student_states.py}`。已有许多 `docs/student_*` 和 `scripts/php_student/*` 在当前工作树中尚未纳入 Git；接手时先检查 `git status`，不要误删或把失败 checkpoint 当成正式产物提交。
