# Student 训练分布阶段覆盖探针（2026-10-27）

## 方法

从正式 Student `model_29999.pt` 加载完整策略/Teacher 权重及其内嵌训练配置，沿用原 motion bundle、随机初始相位、随机化和 `add_onpath_obstacle=true, num_variants=3` 的地形预处理。只将并行环境数从 2048 降到 128，运行 500 个无权重更新的 Student 随机动作控制步，共 64,000 个环境步。按 reference 相位每 20% 分组，统计 Teacher 标签有效率、仅在有效标签上的 Student–Teacher action-mean L2、reference 位置误差。

| reference 相位 | 样本数 | Teacher 标签有效率 | 有效样本动作 L2 | reference 位置误差均值 |
|---|---:|---:|---:|---:|
| 0–20% | 6,431 | 99.2% | 0.725 | 0.122 m |
| 20–40% | 13,969 | 56.5% | 1.401 | 0.305 m |
| 40–60% | 13,095 | 44.6% | 0.979 | 0.472 m |
| 60–80% | 12,837 | 92.4% | 1.110 | 0.204 m |
| 80–100% | 17,668 | 76.8% | 0.690 | 0.314 m |

## 判断

20–60% reference 相位恰是 full-clip 失败前的困难阶段，Teacher 标签有效率从起始阶段的 99.2% 降到约 45–56%。20–40% 的有效样本动作差也最大。这说明正式训练的全局 behavior loss 可以被大量较容易的相位稀释，而困难阶段同时存在**模仿误差高**和**标签被 expert-lost mask 丢弃**的问题。仅增加总 iteration 未必能有效增加可学习的关键样本。

这不是原 30k 训练的历史逐步计数，而是其**保存配置 + 末期 checkpoint**的 128-env、500-step 推断式训练分布探针；不能据此声称整个训练历程的有效率都是上述数值。Teacher 标签失效保护是官方设计，不能直接取消 mask 并把零动作当示范。单个 seed 的聚合结果也不能定位到某一条 motion 或某个 obstacle variant。

## 下一步最小对照

保持 Student 网络、Teacher、PPO/DAgger 权重和总样本预算不变，仅改变采样/roll-in：提高 20–60% 相位中**仍有有效 Teacher 标签**的障碍入口状态覆盖；在状态偏离前短时 Teacher roll-in，再交还 Student。先做短预算 A/B，必须同时记录各相位有效标签数和动作 L2，最终用无接管、独立 seed 的 clean full-clip 成功率验收。若标签覆盖提高而成功仍无改善，再优先审查 depth 几何/时延与 velocity command 的信息是否足够。

## 复核

- 原始统计：`reports/php_student/train_distribution_phase_probe_seed42.json`。
- 入口：`scripts/php_student/run_student_phase_probe.sh`；实现：`scripts/php_student/probe_student_train_distribution.py`。
- 无接管逐帧对照：`docs/student_paired_trace_pilot_2026-10-27.md`。
