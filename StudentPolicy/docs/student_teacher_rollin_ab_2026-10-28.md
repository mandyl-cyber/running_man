# Student 障碍前 Teacher roll-in A/B

目标：检验 Student 在障碍过渡段的差表现，是否主要来自自身控制导致的状态分布偏移、随后 Teacher 标签失效。此实验不是重新训练 Teacher，也不是把训练评估中的 Teacher 接管算作 Student 成功。

## 设计

- A/B 均从同一 `model_29999.pt` 恢复，保持原保存配置、PPO/DAgger 参数、运动库、环境数和随机种子一致。
- 更新后的相位窗口为 25%–55%（障碍接近和过渡）；两组均将此段 PPO gate 设为 0，使用 DAgger 项；其余相位保留原 gate。这样 Teacher 执行的动作不会被当成 Student 的 on-policy PPO 样本。
- A 组全程由 Student 驱动；B 组只在上述窗口且 Teacher 标签有效时由 Teacher 驱动，窗口外由 Student 驱动。差别仅是这段环境控制动作。
- 首先采集无梯度 probe，比对 40%–60% 困难相位的有效 Teacher 标签比例；B 至少提高 5 个百分点，且至少 25% 的合格轨迹各提高 5 个百分点（每组每轨迹该相位至少 100 样本），才允许短训。probe 是机制筛选，不是统计显著性证明。
- 两组各续训 100 iteration 后，在独立 seed=43、clean、40 clips × 3 episodes 下进行 deterministic frame-0 full-clip 评估。评估强制 `takeover=none`，比较总体成功率及逐 clip 胜负。

## 命令

在 `/home/sjtu/running_man/StudentPolicy`：

```bash
bash scripts/php_student/run_teacher_rollin_ab.sh preflight
bash scripts/php_student/run_teacher_rollin_ab.sh diagnose
bash scripts/php_student/run_teacher_rollin_ab.sh train
bash scripts/php_student/run_teacher_rollin_ab.sh eval
bash scripts/php_student/run_teacher_rollin_ab.sh summary
```

也可对一个全新输出目录执行 `bash scripts/php_student/run_teacher_rollin_ab.sh all`。默认输出在 `reports/php_student/teacher_rollin_ab_v3/`；重新实验需设置新的 `STUDENT_ROLLIN_ROOT`，脚本拒绝覆盖已有结果。`diagnose` 不需预检通过，会独立评估源 Student 的失败帧。可用 `STUDENT_ROLLIN_NUM_ENVS`、`STUDENT_ROLLIN_PROBE_ROLLOUTS`、`STUDENT_ROLLIN_ITERATIONS` 调整预算，但两组始终同配。

## 判读与局限

若 B 在 probe 中有效标签覆盖提升，且独立无接管评估也优于 A，才支持“roll-in 缓解状态分布偏移”；只改善标签覆盖而不改善最终成功，说明还存在可观测性、动作模仿或恢复控制问题。若 probe 不提升，脚本停止，不盲目续训。无论结果如何，该 A/B 并未直接证明普通行走不退化，也未跨多个训练 seed；正结果后需要扩大种子与正常步行评估。训练过程中的有效标签比例和最终 full-clip 成功率不能互相替代。

第一次 A/B（12%–25% 接管）预检有效标签只提高 0.31 个百分点，却因旧门槛“>0”而续训；最终两组独立 seed43 评估均为 0/120。新代码用 5 个百分点和逐轨迹门槛阻止这类误放行。第二次 12%–25% 预检得到 +2.18 个百分点，在新门槛下已正确停止。源 Student 的 seed43 无接管诊断为 0/120，失败进度中位数 53.4%，120 次失败中 113 次位于 40%–60%，没有一次在 12%–25%。因此把下一轮接管窗口改为 25%–55%，先预检，不直接续训。

## 25%–55% 实测结果（v3）

- 预检：40%–60% 有效标签由 53.5% 升至 96.6%（+43.1 个百分点）；79 条样本充足的训练 clip 中 61 条提升至少 5 个百分点；40%–60% Teacher 实际接管 19,298 步，门槛通过。训练 bundle 的 clip 数为 100，正式评估子集为 40，不能把两者混称。
- 两组均从同一 `model_29999.pt` 续训 100 iteration，保存 `model_30098.pt`。训练期 40%–60% 有效标签：Student 对照 70.6%，Teacher roll-in 96.2%。
- 独立 seed43、clean、40 条 motion × 每条 3 回合、无 Teacher 接管：A 为 **0/120**，B 为 **6/120（5%）**，6 次成功分别落在 6 条 clip。失败进度中位数 A 45.9%，B 50.0%。两份报告 `complete=true`。
- 解释：对齐困难相位的 Teacher roll-in 既明显提高训练标签可用性，也产生初步无接管成功；但 5% 远未达到可用或验收标准。不能仅用这一个 seed、100 iteration 断言稳定泛化，也不能把训练期标签有效率当作部署性能。

## 独立评估 seed 44–46 复核

为确认 seed43 的 6/120 是否偶然，对原始 `model_29999.pt` 与 Teacher-roll-in `model_30098.pt` 做 seed44、45、46 的同条件配对评估：clean、40 motions × 3 episodes、无接管；所有报告 `complete=true`。seed43 原始 checkpoint 的结果来自同 checkpoint 的 v2 source 评估。

| Eval seed | 原始 checkpoint | Student-only continuation | Teacher-roll-in continuation |
|---:|---:|---:|---:|
| 43 | 0/120 | 0/120 | 6/120 (5.0%) |
| 44 | 0/120 | 0/120 | 6/120 (5.0%) |
| 45 | 0/120 | 未测 | 4/120 (3.3%) |
| 46 | 0/120 | 未测 | 9/120 (7.5%) |
| 合计 | **0/480** | 0/240 | **25/480 (5.2%)** |

Roll-in 的非零成功分布在 16 条不同 motion；没有 motion 在四个 seed 中都成功。故“roll-in 能带来少量成功”在评估 episode 数上重复出现，但具体成功轨迹不稳定，整体成功率仍仅 5.2%，95% 失败。**判定：机制信号复现，但性能 gate 未通过；不晋升、不扩长训练，也不称为有效 Student。** Student-only continuation 在已测的 seed43/44 仍为 0/240。

seed44–46 报告位于 `reports/php_student/teacher_rollin_ab_v3/eval/confirm_seed{44,45,46}/{source,teacher}/full_clip_seed{seed}.json`。本轮只做评估，没有新增训练或修改 checkpoint。

下一步先做候选模型的阶段分解诊断：对比 20–60% 相位有效 Teacher 状态上的 Student–Teacher action L2、各 DOF 误差与 rollout 首次偏差帧，并按成功/失败 motion 对照。之后只选一个训练改动（优先连续片段级而非逐步随机切换的 roll-in），短训后必须重过相同的无接管多 seed gate；不应继续只凭 coverage 上升扩展训练。
