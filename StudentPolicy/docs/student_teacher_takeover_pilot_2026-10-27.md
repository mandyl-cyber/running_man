# Student–Teacher 动作接管小规模验证（2026-10-27）

## 问题与设计

官方 PHP Student checkpoint `model_29999.pt` 在固定 seed=42、clean、40 条 motion 的 frame-0 full-clip 评估中为 0/200；相同数据上的 Teacher 可完成 183/200。为了验证失败是否主要来自 Student 在障碍接近阶段的动作，而不是环境、轨迹或成功判定，在**同一个 Student 评估环境**中保持观测、物理、终止和计数逻辑不变，只替换动作：从每条 clip 的 25% reference 帧开始，连续用 checkpoint 内 Teacher 的动作接管 50、100 或 200 个控制步（约 1、2、4 秒），随后交还 Student。另以全程 Teacher 接管作为阳性对照。

评估采用同一 Student checkpoint、40 条 motion、clean、seed=42、每条 3 次；全程接管对照每条 1 次。结果 JSON 均有 `complete=true`。窗口是**reference 相位窗口**，并非基于真实接触事件的精确触发。

| 动作来源 | 成功/有效回合 | 成功率 | 至少成功一次的 clip | 忽略的 simulator timeout |
|---|---:|---:|---:|---:|
| Student 全程（先前基线，40×5） | 0/200 | 0% | 0/40 | 0 |
| Teacher 接管 50 步 | 57/120 | 47.5% | 28/40 | 38 |
| Teacher 接管 100 步 | 77/120 | 64.2% | 32/40 | 52 |
| Teacher 接管 200 步 | 91/120 | 75.8% | 37/40 | 59 |
| Teacher 全程（阳性对照，40×1） | 40/40 | 100% | 40/40 | 0 |

## 判断

在 Student 环境中，全程 Teacher 接管 40/40，排除了“Teacher 权重/路由不可用”以及这批 clip 在当前环境必然无法完成的解释。仅在早期 reference 相位短时替换动作，就出现随接管时长上升的明显成功率提升。这为**Student 在接近/准备越障阶段的闭环动作选择不足**提供了干预证据；原来 0% 不能简单归咎于 success 判定过严，也不能只归咎于轨迹不可执行。

此试验**不能**区分深度感知、历史信息、蒸馏目标或训练分布哪一项导致动作不足。不同组虽然 seed 相同，但非严格逐回合配对；各组有被忽略的 simulator timeout，成功率分母是达到终止/完成条件的有效回合。25% 起点也未与实际 obstacle contact 对齐。因此这些是定位性证据，不是泛化性能的最终置信区间或正式验收指标。

## 下一步

在同一套 40 条 clip 上保存失败前后的 Student/Teacher 动作差、深度图和 root/足端状态；按接触事件对齐，而不只按 reference 百分比。随后做一个控制变量实验：保持 Student 架构与训练预算不变，只强化障碍前/接触阶段的 Teacher action supervision（或短时 DAgger 式 roll-in），用独立 seed 的无接管 full-clip clean 评估验证改善。不得把接管后的成功率当作 Student 自身成功率。

## 可复核文件

- 评估代码：`scripts/php_student/eval_student_full.py`，启动脚本：`scripts/php_student/eval_student_full.sh`。
- Student 基线：`reports/php_student/paired_seed42_student5/full_clip_seed42.json`。
- 50/100/200 步：`reports/php_student/takeover_window50_seed42_3/full_clip_seed42.json`、`reports/php_student/takeover_window100_seed42_3/full_clip_seed42.json`、`reports/php_student/takeover_window200_seed42_3_gpu/full_clip_seed42.json`。
- 全程 Teacher：`reports/php_student/takeover_teacher_all_seed42_1/full_clip_seed42.json`。
