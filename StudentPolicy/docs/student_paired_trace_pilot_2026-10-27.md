# Student 障碍阶段逐帧配对诊断（小样本）

## 设置

在官方 PHP Student `model_29999.pt` 的 clean、seed=42、40-motion 评估中，每条跑 1 次；选 5 条（直行、左 45°、右 45°、右 90°）保存首次 episode 的逐控制步记录。每步记录同一状态下的 Student 与 Teacher 动作、actor/teacher observation、reference/机器人位置、root state、左右足接触力；每 5 步保存一张 58×87 深度图。**Teacher 动作仅作诊断，没有接管控制。**

评估报告完整：0/40 success。以下“动作差起点”定义为 Teacher 有效时，20 步移动平均的 29 维 action L2 首次超过 1.0；它是人为诊断阈值，不是物理失败判定。约 50 控制步为 1 秒。

| motion ID | 动作差起点帧 | Teacher 首次失效帧 | 失效触发项 | 当时 reference 位置误差 | 最终终止帧 |
|---:|---:|---:|---|---:|---:|
| 0（直行） | 181 | 223 | tracked body position | 0.438 m | 415 |
| 10（左 45°） | 121 | 238 | tracked body position | 0.465 m | 415 |
| 13（左 45°） | 217 | 253 | tracked body position | 0.324 m | 478 |
| 25（右 45°） | 180 | 219 | reference position | 0.508 m | 362 |
| 34（右 90°） | 204 | 239 | tracked body position | 0.375 m | 448 |

深度数组在这 5 条记录中均为有限值；左右足接触力、逐帧 observation 与动作数组也已保存。新记录精确拆分了 expert-lost 条件：4/5 是指定 ankle/wrist tracked-body error >0.5 m，1/5 是 reference position error >0.5 m，没有一条由 orientation 阈值触发。Student–Teacher action-L2 的 20 步均值在 Teacher 标签失效前就超过 1.0；该起点 frame 的 reference anchor 误差仅 0.099–0.247 m。

深度拼图（每条第 100/150/200/250 帧）显示障碍轮廓在动作分歧附近可见并随运动变化。后续追踪对齐审计已进一步定量检查相机时延与静态地形几何（见下文）；足端净接触力仍不能单独区分地面与障碍物接触。

## 解释及限制

动作分歧发生在 Teacher 标签失效之前。PHP 的 `which_motion` 会在 anchor、orientation 或指定身体位置误差超过阈值时将路由设为 -1；此后 `teacher_act` 输出零动作，DAgger loss 将该样本 mask。因此这里的 Teacher 首次失效**不是 Teacher 真实输出零动作**，也不能把失效之后 Student–Teacher action L2 当作模仿误差。失效时的 reference 位置误差为 0.324–0.508 m，远早于最后约 1 m 的 tracking 终止。

这支持两段式失效链：Student 较早与可用 Teacher 动作分歧 → 状态偏离到专家标签无效区 → 随后累积误差并终止。它**不能确定**第一段分歧来自深度感知、命令信息不足还是策略学习不足；也不能证明训练时完全没有相关有效样本。此前 50/100/200 步 Teacher 接管成功率依次为 47.5%/64.2%/75.8%，与“早期动作很关键”相符。

## 下一步判别实验

1. 对 5 条记录中动作差起点前后约 2 秒的深度图做人工几何检查：障碍是否进入视野、位置与相对距离是否匹配机器人/地形；确认深度延迟是否使关键动作来不及执行。
2. 在训练 rollout 中按 reference 帧与障碍接触阶段统计 `which_motion` 有效比例、DAgger 样本数及有效样本上的 Student–Teacher action L2。仅看全局平均 loss 无法回答这些问题。
3. 若观测正确而关键阶段动作差仍大，再做预算和 seed 固定的“障碍前有效样本加权/重采样”小规模蒸馏对照；用**无接管**、独立 seed 的 full-clip success 验证，而非接管成功率。

## 产物

- 评估：`reports/php_student/paired_trace_seed42_5clips_retry/full_clip_seed42.json`。
- 最新完整评估：`reports/php_student/expert_loss_audit_seed42_v1/full_clip_seed42.json`，0/40 success。
- 最新逐帧数组：同目录 `traces/motion_00.npz`、`motion_10.npz`、`motion_13.npz`、`motion_25.npz`、`motion_34.npz`。
- 最新深度拼图：`reports/php_student/expert_loss_audit_seed42_v1/depth_montage.png`。
- 采集入口：`scripts/php_student/eval_student_full.sh`；摘要及拼图工具：`scripts/php_student/summarize_paired_traces.py`、`scripts/php_student/render_paired_depth.py`。

## 追踪对齐追加审计（2026-10-28）

对同 5 类轨迹重新做 1 episode/clip 的小评估，并记录深度缓冲区中的相机位姿、随机 latency 与被选图像。它是定位实验，不替代完整 40-clip 统计。

- 速度命令：5/5 clips 的每一帧都与源 NPZ `vel_cmd[frame]` 精确相同；整体 MAE、逐帧最大绝对误差均为 `0`。因此没有证据表明 15-D command 使用了错帧或错轨迹。
- 图像时延：记录到的 latency 只有 3/4 帧，符合 PHP 配置 `(3,4)`；按相机缓冲区重建的图像与实际 student 输入逐像素匹配，最大/平均误差均为 `0`。采集的相机位置、四元数随所选缓冲帧同步。
- 几何：将输入图像反归一化后，按 D435i 的 89.5° HFOV、裁剪/缩放和记录的相机位姿，对预处理生成的地形 mesh 做射线重投影。约 38.6 万个有静态地形命中的采样像素中，88.2% 深度差小于 0.10 m、90.4% 小于 0.20 m，中位绝对误差 0.029 m。mesh-only 重投影不包含机器人自身动态网格；深度处理还有孔洞、边缘置换、噪声和缩放，因此剩余误差不能直接算作标定错误。
- 这轮 5-clip 小评估依旧为 0/5 成功；trace 中动作差起点早于 expert-lost，深度保持有限。

结论：目前足以排除明显的 velocity-command 帧错位、深度缓冲误取帧，以及静态地形投影的重大几何偏差；**不足以据此直接改一个“深度 bug”并声称会修复 Student**。更符合证据的下一步是定位训练数据/优化是否覆盖了动作开始分歧的帧段：在固定状态与地形的配对观测上比较 Student/Teacher 动作，按 motion phase 汇总有效 expert 标签数及 action L2，再仅对障碍前—接触—越障过渡做小规模采样/权重对照。当前没有启动训练，也没有据此改训练超参数。

复现入口：

```bash
/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python \
  scripts/php_student/audit_student_trace_alignment.py \
  reports/php_student/expert_loss_audit_seed42_depth_alignment_v2/traces \
  /home/sjtu/.cache/php-parkour/motions-low-climb-76

/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python \
  scripts/php_student/audit_depth_geometry.py \
  reports/php_student/expert_loss_audit_seed42_depth_alignment_v2/traces \
  reports/php_student/expert_loss_audit_seed42_depth_alignment_v2/terrain_mesh_seed42.npz \
  --frame-stride 4
```

新评估产物：`reports/php_student/expert_loss_audit_seed42_depth_alignment_v2/`（5-clip report、trace、地形 mesh）。采集相机位姿及 latency 的改动位于 `scripts/php_student/eval_student_full.py`。

## 下一轮可复现命令：逐项定位 expert-lost

新版 trace 会额外保存 PHP `which_motion` 的三个原始失效条件、路由 index、15-D velocity command 和实际输入深度。使用一个全新的输出目录；本评估会检查完整 40 条 clip，每条 1 回合，但只为下列 5 条保存逐帧数组：

```bash
cd /home/sjtu/running_man/StudentPolicy

STUDENT_CHECKPOINT=/home/sjtu/php_parkour_official/logs/WBT-Holosoma/20261018_172554-local-student-official-seed42-locomotion/model_29999.pt \
STUDENT_EVAL_SEED=42 \
STUDENT_EVAL_OBSTACLE_SEED=42 \
STUDENT_EVAL_EPISODES_PER_MOTION=1 \
STUDENT_EVAL_MAX_STEPS=5000 \
STUDENT_EVAL_DIAGNOSTICS=1 \
STUDENT_EVAL_TRACE_MOTION_IDS=0,10,13,25,34 \
STUDENT_EVAL_TRACE_DEPTH_STRIDE=5 \
STUDENT_EVAL_REPORT_DIR=reports/php_student/expert_loss_audit_seed42_v1 \
bash scripts/php_student/eval_student_full.sh

/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python \
  scripts/php_student/summarize_paired_traces.py \
  reports/php_student/expert_loss_audit_seed42_v1/traces

/home/sjtu/.holosoma_deps/miniconda3/envs/php/bin/python \
  scripts/php_student/render_paired_depth.py \
  reports/php_student/expert_loss_audit_seed42_v1/traces \
  reports/php_student/expert_loss_audit_seed42_v1/depth_montage.png
```

**先检查 summary 中首次失效项和 depth montage，再决定是否启动 roll-in 训练。**此命令不接管动作、不更新权重；若输出目录已存在，请将 `STUDENT_EVAL_REPORT_DIR` 改成新目录名。
