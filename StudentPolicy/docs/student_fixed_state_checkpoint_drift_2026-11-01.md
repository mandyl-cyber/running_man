# Student 同状态 checkpoint 漂移诊断（2026-11-01）

## 问题与方法

固定最佳 Student `model_30098.pt` 的评估 trace，不重新跑仿真；在**完全相同的 actor observation 与精确对齐的深度图**上，比较原模型、普通续训第 25 次和第 50 次 checkpoint 的确定性动作。Teacher action 是同一状态的有效标签。两种状态分布分别统计：Student 自主 rollout，以及 Teacher 全程控制 rollout。取 40 条 low-climb-76 轨迹，按原 Student trace 中首次连续 10 帧 `||a_student−a_teacher||₂>1` 的帧对齐。

只使用有实际记录深度图的帧，不能把稀疏深度帧线性插值成逐控制步观测。原 checkpoint 的离线预测与 trace 中记录的 Student 动作最大绝对差为 **2.4×10⁻⁶**，因此模型加载和输入对齐检查通过。每组误差为逐样本 `||a_student−a_teacher||₂` 的均值；数值越低越好。

| 状态来源／相对首次分歧窗口 | 有效深度帧 | 原模型 | 第 25 次 | 第 50 次 |
|---|---:|---:|---:|---:|
| Student 自身／−80 至 −20 帧 | 448 | 0.675 | 0.684 | 0.689 |
| Student 自身／−20 至 0 帧 | 160 | 0.762 | 0.773 | 0.778 |
| Student 自身／0 至 +20 帧 | 160 | 1.885 | 1.892 | 1.878 |
| Student 自身／+20 至 +80 帧 | 304 | 1.366 | 1.381 | 1.385 |
| Teacher 轨迹／−80 至 −20 帧 | 448 | 0.479 | 0.493 | 0.503 |
| Teacher 轨迹／−20 至 0 帧 | 160 | 0.497 | 0.514 | 0.520 |
| Teacher 轨迹／0 至 +20 帧 | 160 | 0.718 | 0.726 | 0.738 |
| Teacher 轨迹／+20 至 +80 帧 | 473 | 0.646 | 0.693 | 0.725 |

Teacher 轨迹的准备窗口中，第 50 次相对原模型有 **35/40** 条轨迹误差增加；分歧后窗口为 **36/40**。在 Student 自身状态的准备窗口，增加的是 **29/40**。三个定点案例的首次持续分歧帧分别为 clip 4/12/24 的 **226/230/182**；其准备窗口误差在原模型上已经约 **0.850/0.716/0.695**，第 50 次为 **0.855/0.712/0.723**。第 50 次在这些固定状态上没有出现与 full-clip 归零同量级的动作误差爆炸。

## 解释和边界

1. **排除一个简单解释：**第 50 次 checkpoint 并非因模型加载错位或离线动作整体崩溃而得到 0/40。最佳模型在 Student 自身的关键状态上本就存在较大 Teacher–Student 动作差；续训带来较小但广泛的动作漂移。
2. **更符合现有证据的工作假设：**较小的动作变化在接触过渡的闭环动力学中被放大，而当前训练优化没有稳住原有的完整轨迹能力。Teacher 轨迹上的误差普遍上升与此前 PPO 梯度相对 DAgger 更强的观测一致，但本实验不能证明 PPO 是唯一原因。
3. **不能过度推断：**Teacher 驱动与 Student 驱动的同名 reference 帧不是同一物理状态；表中两种状态来源不能相减作为因果效应。深度每约 5 个控制步记录一次。这里只有固定状态动作误差和此前 seed43 的 3/40→0/40 闭环结果，没有重建第 25/50 次的完整逐帧自主轨迹，也不能由此判定视觉不可辨或某个关节必然出错。

## 下一步

先不要继续调 DAgger 倍率或延长训练。下一次最有判别力的实验是对第 25 和 50 次 checkpoint 做**相同 seed、逐轨迹配对的完整自主 rollout trace**，定位哪几帧的小动作差开始转成接触、pelvis 和失败边界差；随后在那个窗口核对实际 storage 中的 PPO advantage、DAgger 梯度及有效标签。若固定状态误差小、闭环分叉明显，应优先处理 Student 偏离状态和恢复训练，而不是把单帧 MSE 当作最终目标。

产物：[summary.json](../reports/php_student/continuation_collapse_v1/fixed_state_drift_all40_v1/summary.json)、[per_frame.csv](../reports/php_student/continuation_collapse_v1/fixed_state_drift_all40_v1/per_frame.csv)；复现脚本：[analyze_student_checkpoint_drift.py](../scripts/php_student/analyze_student_checkpoint_drift.py)。
