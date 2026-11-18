# Student 困难相位有效标签加权 A/B（2026-10-27）

## 目的与控制变量

检验“只加强 Teacher 仍有效的障碍入口样本监督”能否改善无接管 full-clip。两组均从官方代码训练得到的 Student `model_29999.pt` 及其 optimizer 状态继续；同一保存配置、motion bundle、三种 obstacle variants、随机化、seed=42、256 env、200 iterations（每组约 122.9 万环境步），网络和 PPO/DAgger 配置相同。基线组每个有效标签权重 1；实验组仅把 reference 相位 20–60% 且 Teacher 路由有效的 DAgger per-sample loss 乘 2。expert-lost 样本仍然被 mask；Teacher action、PPO、终止和环境不改。两组各先通过 2-iteration 预检，再独立完成正式短续训。

这是一项**自定义诊断干预**，不是官方原始 recipe 的成功复现声明；两组使用相同代码路径，唯一实验开关是上述权重。训练仍使用 256 env 而非正式训练的 2048 env，因此只能作为小预算方向筛选。

## 无接管验收（独立 seed=43）

clean、40 clips、每条 3 次，固定 seed=43 和 obstacle seed=43；原 checkpoint 也用同条件补测。

| checkpoint | full-clip 成功 |
|---|---:|
| 原始 `model_29999.pt` | 0/120 |
| 续训基线 `baseline/model_30198.pt` | 0/120 |
| 有效标签加权 `focus/model_30198.pt` | 0/120 |

三份报告均 `complete=true`。因此**该小预算加权方法未通过成功率 gate**，不得把它当成 Student 改善版本。

## 阶段探针（同保存训练配置，各 128 env × 500 步）

| reference 相位 | 基线有效率 | 加权有效率 | 基线有效动作 L2 | 加权有效动作 L2 |
|---|---:|---:|---:|---:|
| 0–20% | 99.4% | 99.0% | 0.805 | 0.757 |
| 20–40% | 55.7% | 54.6% | 1.478 | 1.362 |
| 40–60% | 53.7% | 53.4% | 1.088 | 1.086 |
| 60–80% | 87.6% | 86.9% | 1.155 | 1.190 |
| 80–100% | 80.1% | 75.6% | 0.747 | 0.772 |

20–40% 的有效标签 action L2 下降约 7.8%，说明加权至少对局部模仿误差有影响；但它没有改善 40–60% 的标签覆盖或全轨迹成功率，并可能伴随其他相位退化。这里的探针是末期 checkpoint 的推断式抽样，不是原训练历史计数；单 seed 差异不可作为统计显著性的证据。

## 经验与下一步

“把已经有效的标签 loss 加倍”不能凭空产生掉出 expert-valid 区域后的可用恢复样本，也不能补 Student 可能缺少的地形/时序信息。**不要据此延长该加权组训练**。下一步优先做更有区分力的控制实验：从障碍前有标签的状态短时 Teacher roll-in，再交给 Student，记录是否因此扩大有效标签覆盖；同时核对深度距离/时延。若覆盖没有提高，回到观测与数据生成检查；若覆盖提高但无接管成功率仍为零，再检查动作目标与策略结构。任何新方法仍须与原 checkpoint/普通续训在相同独立 seed 下比较。

## 文件

- 训练代码：`scripts/php_student/train_student_phase_ab.py`、`run_student_phase_ab.sh`。
- A/B checkpoints 与合同：`reports/php_student/phase_ab_main_v1/{baseline,focus}/`。
- 评估：同目录 `eval_source_seed43/`、`eval_baseline_seed43/`、`eval_focus_seed43/`。
- 阶段探针：同目录 `probe_baseline_seed42.json`、`probe_focus_seed42.json`。
