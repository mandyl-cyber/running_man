# Student 诊断计划执行记录（2026-10-28）

> **更正（同日）：**第 6 节旧标签冲突探针误删 `teacher_obs[0]` 而非路由列 `[449]`，且将失败 trace 的末帧错当完整 reference 长度。该节的 5,539 样本、近邻数值和“没有严重冲突证据”的判断已失效；应以 [student_true_phase_reaudit_2026-10-28.md](student_true_phase_reaudit_2026-10-28.md) 的重算结果为准。历史结果保留以便审计，不应继续引用为当前结论。

## 合同与结论

对象：正式 Student `model_29999.pt`；5 条代表性 low-climb motion；clean、seed=42、固定 obstacle seed=42。Teacher 接管只用于对照，不作为 Student 成功。**没有修改原 checkpoint，也没有启动长训练。**

主要结论：失败不是单纯的严格 tracking 指标误判，也未发现速度指令错帧、Teacher 路由/标签、DAgger mask 或深度缓冲的明显实现错误。更强的证据指向：Student 在仍处于有效专家状态时，障碍过渡段动作已偏离；闭环运行随后使有效标签消失，训练可用的困难状态覆盖降低。小样本纯监督能显著拟合这些动作，故不能说网络“学不了”；也尚不能把全部责任唯一归于样本覆盖或视觉信息。

## 1. 评估语义与实际进度

相同 5-clip 小评估：Teacher 全程控制 5/5 full-clip 完成；Student 自主控制 0/5。另从本次实际地形 mesh 的障碍盒和 reference 起终点构造**沿参考路径的 XY 投影代理量**：5 条 Student rollout 的最大 anchor 进度都超过第一障碍前缘，但都没超过第一障碍中心；Teacher 最终完成全程。因此当前失败不只是“过了障碍但被严格 tracking 阈值判负”。投影不能证明接触、机身净空或物理越障成功，正式任务指标仍应另外定义。

| motion | Student 最大投影 | 第一障碍中心投影 | Student 是否越过中心 |
|---|---:|---:|---|
| forward 0.6/0.7 | 1.556 m | 1.963 m | 否 |
| left45 0.9/0.7 | 1.379 m | 2.031 m | 否 |
| left45 1.2/1.2 | 1.789 m | 2.108 m | 否 |
| right45 0.6/1.2 | 1.384 m | 2.087 m | 否 |
| right90 0.9/0.7 | 1.424 m | 1.660 m | 否 |

## 2. 监督链路核对

使用真实 checkpoint 和深度帧对齐的保存状态，离线重算 40 个 Teacher 有效样本及 40 个 expert-lost 样本；**不执行 optimizer step**。

- 路由列为 `teacher_obs[449]`；29-D action 顺序与保存的 29 个 G1 DOF 配置一致。重算 Teacher 动作相对 trace 最大绝对误差 `1.2e-6`；Student 动作 `5.3e-5`（数值级别差异）。
- PHP 生产版 DAgger per-sample loss 与独立公式逐项完全一致；40 个失效样本的 DAgger loss 全为 0。Teacher 参数梯度为 0；Student 与 depth backbone 梯度范数分别为 `0.413`、`0.032`，均有限且非零。
- 官方集成/深度导出相关 CPU 测试 9/9 通过。原训练 TensorBoard 最末 `ppo_coef=0.9`、`Loss/behavior≈0.0117`、`ppo_gate_mean≈0.611`。全局 behavior loss 会被 mask 与容易相位稀释，不能当困难阶段模仿成功的证据。

这些核对排除了明显的标签路线/屏蔽/反向传播断路；它们不是原 30k 训练历史中每个 batch 的重放，也未核对所有接触动力学。

## 3. 同状态配对动作对照

在每个控制步同时计算 Student 与 Teacher 动作，然后分别让 Teacher 或 Student 驱动仿真。Teacher 驱动轨迹中路由全程有效且 5/5 成功；即使处在这些正常专家状态，第 200–300 帧的 Student–Teacher 29-D action L2 均值仍为 `0.852–1.191`。Student 自己驱动时，同一帧段内仍有效的标签上 L2 为 `1.446–2.792`；约第 224–253 帧开始 expert-lost，最终在第 387–475 帧 tracking 失败。共同较大误差列包括两侧 knee、hip pitch 和 shoulder pitch（DOF 映射来自保存配置）。

这是两种**不同闭环轨迹**，不能把对应帧视作完全同一物理状态；“同状态”仅指每条 rollout 的同一步同时产生两套动作。结果显示分歧在正常专家状态就存在，闭环偏离又将其放大。

## 4. 训练缓冲区覆盖

从保存的训练配置和 Student checkpoint 建立 512-env 环境，运行 21 个真实 `_rollout_step`（每个 24 步，共 258,048 条进入 DistillationPPO storage 的 transition），不更新权重。采集前的 Teacher action 与 storage 中的 label 最大误差为 0。另以 128-env 重复同类探针，方向一致；下表以覆盖更充分的 512-env 数据为准。

| reference 相位 | storage 样本 | 有效 Teacher 标签比例 | 有效样本 action L2 |
|---|---:|---:|---:|
| 0–20% | 19,050 | 99.5% | 0.715 |
| 20–40% | 52,652 | 56.1% | 1.395 |
| 40–60% | 60,226 | 54.7% | 0.973 |
| 60–80% | 57,094 | 89.0% | 1.138 |
| 80–100% | 69,026 | 76.1% | 0.712 |

这与此前 500 步推断探针方向一致。分 motion × phase 统计也已保存：5 条代表轨迹的 20–40% 有效率为 37.5–57.5%，40–60% 为 33.4–85.6%，说明聚合均值会掩盖轨迹差异。注意：这是**末期 checkpoint + 重新初始化的训练环境**，不是原始 30k 训练的历史标签计数；探针有效样本约 18.5 万，低于 PPO gate 的 50 万样本 warm-up，故此探针 gate 为 1.0，不能与历史末期约 0.611 直接等同。保存配置还设定 `start_at_timestep_zero_prob=0`、`distillation_warmup_steps=0`：训练从随机相位开始且无 Teacher 驱动 warm-up，这是可能的长时序/恢复分布缺口，不是已证实的唯一原因。

## 5. 条件性最小干预：小数据过拟合

在 Teacher 控制的第 180–320 帧，取 4 条 motion 的 92 个训练深度帧，用原 checkpoint 的 Student + depth backbone 做 300 步纯 MSE 诊断；另保留相邻的 20 帧及第 5 条 motion 的 28 帧。**不保存为可部署策略，不声称 full-clip 改善。**

| 集合 | 原 action L2 | 诊断后 action L2 |
|---|---:|---:|
| 92 训练帧 | 0.963 | 0.136 |
| 20 相邻保留帧 | 1.007 | 0.730 |
| 28 帧未训练 motion | 0.792 | 0.720 |

这证明同一网络与优化器至少能拟合小规模有效专家状态；对未训练帧/轨迹的改善有限，不能推断闭环成功。此前“仅把现有有效标签权重乘 2”的独立 seed 短续训仍为 0/120 success，因此不重复该方案。

## 6. Student 闭环状态的跨轨迹标签冲突审计

为检验“Student 看到近似状态时，Teacher 是否给出彼此矛盾的动作”，从 Student 自主运行的 40 条 trace 中取 reference phase 25–65% 且 Teacher 标签有效的 5,539 个样本。每条轨迹等量抽取最多 60 帧，共 2,400 帧；按标准化输入观测找每帧 5 个最近邻，跨轨迹与同轨迹分开统计。比较 Student 可用的 actor 输入、actor+冻结深度特征，以及去掉 routing identity 的 Teacher 特权观测。

| 输入空间 | 跨轨迹近邻 Teacher action L2 | 同轨迹近邻 L2 | 随机跨轨迹动作对 L2 |
|---|---:|---:|---:|
| actor | 1.553（轨迹 bootstrap 95% CI 1.489–1.620） | 1.389 | 2.843 |
| actor + frozen depth | 1.584（1.521–1.651） | 1.434 | 2.843 |
| privileged Teacher obs | 1.371（1.318–1.426） | 1.475 | 2.843 |

**判读：这不构成“Student 输入存在严重、不可约标签冲突”的证据。** actor 跨轨迹近邻误差比同轨迹高约 12%，提示一定程度的跨 motion 差异；但其远低于随机动作对。加冻结深度特征没有缓解差异；而 privileged 输入的跨轨迹动作差异反而低于同轨迹值，未显示“特权输入能显著消解冲突”的预期模式。此前 5 帧历史 probe 的 held-out L2 也从单帧 1.227 恶化到 1.345，因此目前没有证据支持直接加历史堆叠。

该近邻审计属于关联性诊断，不是因果证明：不同 embedding 的距离数值不能互相比较；跨轨迹最近邻也未严格匹配 terrain、相位和接触模式，且 Teacher action 差异本身不等价于动作标签冲突。综合现有证据，首要瓶颈仍更像是困难阶段有效专家状态覆盖不足、闭环偏差累积；不应把问题简单归结为“观测信息不足”。

完整数据：`reports/php_student/own_state_supervision_probe_v1/label_conflicts_v2.json`；可复现脚本：`scripts/php_student/audit_student_label_conflicts.py`。该脚本使用项目 `php` 环境与 PyTorch 精确近邻计算，不需要 sklearn。

## 下一步 gate

上述 Teacher-roll-in A/B 已完成：同一 `model_29999.pt` 起点、同预算 100 iterations，Teacher 在 25%–55% phase 内驱动且只在 Teacher 标签有效时替换动作；窗口内 PPO gate 设为 0。roll-in 将 40%–60% 有效标签比例从 53.5% 提高到 96.6%，但完整无接管评估四个 seed 合计仅 25/480（5.2%），原始 checkpoint 为 0/480；成功分散于 16 条 motion，没有 motion 在四个 seed 均成功。故覆盖率 gate 通过，性能 gate 未通过，不能晋升此 checkpoint。

下一步不再重复这个 A/B，也不立即延长相同训练。先按成功/失败 motion 做 20%–60% 相位分解：比较有效标签上的逐 DOF Student–Teacher action error、第一次动作偏差帧和首次 tracking failure，并对照 seed43–46 中成功的 16 条 motion。再选择一个更连续的片段级 roll-in 或 recovery 数据干预；必须保持同一 source checkpoint 与训练预算，先做覆盖/action-error gate，再用无接管独立 seed full-clip gate。当前不修改成功阈值、不扩成长训，也不把 Teacher 接管回合计作 Student 成功。

## 产物与复现

- 配对结果：`reports/php_student/expert_loss_audit_seed42_depth_alignment_v2/`（Student）；`reports/php_student/teacher_state_action_audit_seed42_v1/`（Teacher 接管）。
- 动作对照：`reports/php_student/teacher_student_state_comparison_seed42_v1.json`；地形进度代理：`reports/php_student/obstacle_progress_seed42_5clips_v1.json`。
- 真实 checkpoint 梯度/掩码：`reports/php_student/distillation_batch_audit_seed42_v1.json`；storage 覆盖：`reports/php_student/storage_coverage_seed42_512env_500step_v1.json`（含 motion × phase）；小样本拟合：`reports/php_student/teacher_state_obstacle_overfit_seed42_v1.json`。
- 对应脚本：`scripts/php_student/{compare_student_teacher_rollouts,audit_obstacle_progress,audit_distillation_batch,probe_student_storage_coverage,overfit_teacher_state_obstacle}.py`；storage 包装命令为 `bash scripts/php_student/run_student_storage_probe.sh`。
