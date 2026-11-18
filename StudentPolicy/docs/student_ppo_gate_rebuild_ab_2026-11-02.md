# Student 续训 PPO gate 重建 A/B（2026-11-02）

## 问题

最佳 Student `model_30098.pt` 的 checkpoint 不含 motion command 的 `_gate_bin_sum/_gate_bin_cnt/_gate_seen`；普通续训会从冷 gate 重新开始。原训练末期 `ppo_gate_engaged=1`、表均值约 0.649、标准差约 0.176，而先前 50 次普通续训一直 `engaged=0`。因此这是一个真实的**训练状态恢复缺口**，但此前仅把 gate 均值固定为 0.5 仍得到 0/40，不能推断恢复逐相位表必然有效。

## 方法与实现

从同一原 checkpoint 用 256 env、冻结参数、按原 Student 策略采集 500,000 个有效 Teacher 标签，重建 residual EMA gate；没有 optimizer step。重建表覆盖 98.6% bins，均值 0.645、标准差 0.181，统计量接近原训练末期，但**不是原始 gate 表的逐 bin 精确恢复**。保存一次重建状态后，两组从同一 checkpoint、同一训练 seed、同一份 gate 状态开始 50 次训练：A 在训练前清空表，B 保留表。两组合同仅 `gate_rebuild_mode` 不同，25/50 次 checkpoint 均保存。

实现：[训练入口](../scripts/php_student/train_student_phase_ab.py)、[启动脚本](../scripts/php_student/run_student_phase_ab.sh)。预检中发现无更新采样使用 `torch.inference_mode` 会使仿真状态张量不能在下一次 reset 原地更新；已改为 `torch.no_grad`，并增加异常标记与 checkpoint 存在性校验，避免仿真关闭时产生假通过。已保存的 gate 表加载后 1 次训练预检通过。

## 结果与 gate

| 指标 | A：丢弃重建表 | B：保留重建表 |
|---|---:|---:|
| 第 1/50 次 `ppo_gate_engaged` | 0 / 0 | 1 / 1 |
| 第 1/50 次 `Loss/ppo_gate_mean` | 1.000 / 1.000 | 0.616 / 0.654 |
| 第 1/50 次 gate 表标准差 | 0 / 0 | 0.180 / 0.178 |
| seed 43 clean、frame-0、40 clips×1、无接管 | **2/40** | **2/40** |
| 成功 clip ID | 14、15 | 15、21 |

原模型在相同短评估中为 **3/40**（clip 3、32、38）。两组评估报告均 `complete=true`，40 条各 1 回合；B 没保住原来的三条成功轨迹，也没有比 A 增加总成功数。单 seed、单回合不能证明两组成功概率完全相等，但没有满足预设的正效应门槛。

**决策：STOP。** 不从 A/B checkpoint 继续长训，也不做多 seed 扩展评估；最佳 Student 仍是 `model_30098.pt`。本实验把“gate 状态丢失”确认为流程缺口，同时说明**单独重建近似 gate 表不能解决 Student 闭环成功率低或续训退化**。后续不应继续围绕 gate 均值调参。下一项更有信息量的是训练样本覆盖与更新方向：将早期阶段固定比例回放/更新限制和过渡阶段的有效 Teacher 监督分别做单变量短训，仍以闭环成功率而非同状态 MSE 晋级。

证据：[重建统计](../reports/php_student/gate_rebuild_ab_v1/preflight_keep_gpu_v2/baseline/gate_rebuild.json)、[重建状态](../reports/php_student/gate_rebuild_ab_v1/preflight_keep_gpu_v2/baseline/gate_rebuild_state.pt)、[A checkpoint](../reports/php_student/gate_rebuild_ab_v1/control_discard/baseline/model_step_050.pt)、[B checkpoint](../reports/php_student/gate_rebuild_ab_v1/treatment_keep/baseline/model_step_050.pt)、[A 评估](../reports/php_student/gate_rebuild_ab_v1/eval/control_seed43/full_clip_seed43.json)、[B 评估](../reports/php_student/gate_rebuild_ab_v1/eval/keep_seed43/full_clip_seed43.json)。
