# Student 早期动作保持 A/B（2026-11-01）

## 问题与控制变量

最佳 Student `model_30098.pt` 在 clean、无 Teacher 接管的完整轨迹评估中为 39/480；其中 seed 43、每条轨迹 1 次为 3/40。普通续训相同初始 checkpoint、256 env、50 次迭代后为 0/40。此前配对 rollout 观察到：接近障碍时的小动作差会先积累为状态差，随后才在接触区失败。本实验只改变一个训练项：在训练 rollout 中，参考帧 `<100` 的实际状态上，额外约束当前 Student 动作接近冻结的原 Student 动作（动作维度平均 MSE，系数 50）。其他训练设置、初始 checkpoint 与普通续训相同；第 25/50 次 checkpoint 均保存。此约束是**诊断性的能力保持**，不是把原 Student 当作最优 Teacher。

实现位于 [train_student_phase_ab.py](../scripts/php_student/train_student_phase_ab.py) 和 [run_student_phase_ab.sh](../scripts/php_student/run_student_phase_ab.sh)。预检与正式训练的 TensorBoard 均记录了非零 `Loss/anchor_prefix_loss`；正式训练第 1/50 次分别约为 `1.85e-5` / `1.55e-4`，对应 prefix 样本占比约 `12.3%` / `3.2%`。这说明新损失执行了，也提示晚期 rollout 的早期帧覆盖有限。

## 两级 gate 结果

| 检查 | 普通续训 50 次 | 早期保持 50 次 | 结论 |
|---|---:|---:|---|
| 原 Student 状态、深度严格对齐，0–99 帧动作相对原模型的平均 L2 | 0.144 | 0.051 | 漂移减小约 65%，机制 gate 通过 |
| 同上，100–199 帧 | 0.177 | 0.131 | 漂移减小约 26% |
| 同上，≥200 帧 | 0.269 | 0.296 | 漂移反而增大约 10% |
| seed 43、40 clips×1、clean、frame-0、无接管的成功数 | 0/40 | 0/40 | 闭环 gate 失败；原模型为 3/40 |

固定状态审计共使用 40 条原 Student rollout trace 中精确对齐深度图的 3608 帧；原 checkpoint 的动作重放误差低于 `1e-3`。三条原本成功的 clip 3、32、38 在新 checkpoint 上都失败。评估报告完整，40 条轨迹各 1 次；第一次在沙箱中启动因 Warp 深度相机要求 CUDA 而失败，随后在 GPU 环境重跑成功，不能把第一次空报告算作策略结果。

## 判定与下一步

**STOP / 不扩大到多 seed 或长训练。** 这次实验支持“早期动作漂移可以被抑制”，但否定了“仅保持前 100 帧动作就足以保住越障成功率”的简单假设。离线动作距离变小不等于闭环状态与接触时序稳定，且约束窗口外的动作漂移略增。不能由此认定 prefix 约束无用，也不能认定 PPO 是唯一根因；只能说当前配置没有改善端到端能力。

下一次应优先比较同 seed 的新 checkpoint 与原模型在**逐步闭环**中的首次状态分叉、接触相位及前 100 帧实际占比，并检查第 100 帧之后的控制漂移是否接管了失败；若要进一步训练，应另做单变量、短周期、预注册 gate，不能直接加大系数或延长训练碰碰运气。最佳 `model_30098.pt` 仍是当前 Student 基线。

产物： [固定状态审计](../reports/php_student/continuation_collapse_v1/early_anchor_v1/action_drift.json)、[新 checkpoint](../reports/php_student/continuation_collapse_v1/early_anchor_v1/baseline/model_step_050.pt)、[闭环评估](../reports/php_student/continuation_collapse_v1/early_anchor_v1/eval/step050_seed43_gpu/full_clip_seed43.json)、[审计脚本](../scripts/php_student/audit_student_early_anchor.py)。
