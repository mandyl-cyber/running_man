# Student 固定地形配对失败诊断（2026-10-27）

## 实验合同

- Checkpoint：官方 PHP Teacher `model_19999.pt`，本地 Student `model_29999.pt`。
- 同一 40-clip bundle；clean profile；仿真 seed 42；**障碍预处理 seed 42**；每条 1 个障碍变体；每条 5 次。
- Student 使用 proprio + 15 维 velocity command + 延迟 58×87 深度；Teacher 使用 motion reference 与 privileged terrain 信息。因此精确 reference full-clip 是严格诊断，不能单独代表部署任务成功率。
- `bad_tracking` 在 reference 位置误差 >1.0 m、姿态误差或关键身体误差触发时终止。5 次重复共享同一个障碍实例，不是 5 个独立地形。

## 结果

| 指标 | Teacher | Student |
|---|---:|---:|
| frame-0 full-clip 成功 | 183/200 (91.5%) | 0/200 |
| Student 失败触发 | — | reference 位置 199、身体位置 1、姿态 0 |
| Student 失败时 reference 帧进度均值 | — | 53.4% |
| Student 失败时机器人路径投影均值 | — | 29.8% |
| Student 失败时深度图有限值比例 | — | 100% |
| Student–Teacher 动作 L2 均值（有效 Teacher 标签） | — | 0.823 / 29 关节 |

参考原始轨迹与障碍盒几何，首个障碍前缘路径投影均值约 26%，中心约 41%。Student 失败时投影均值约 30%；200/200 次均未超过首个障碍中心。这是**直线路径投影代理量**，转弯轨迹尤其不能把它等同于物理越障成功率。

## 判断及尚未排除的因素

同地形 Teacher 大多能完成，排除了“这批 reference/地形普遍不可执行”作为主要解释。Student 的深度图不是 NaN 或恒定值；但数值有效不证明深度几何、相机标定或时延完全正确。Student 在接近障碍时相对 reference 明显滞后，最终撞上 1 m tracking 阈值；目前证据更像**未掌握障碍进入/爬升动作**，而不只是最后几帧被严格成功定义误判。

仍需区分：深度感知/地形配准、velocity-only 命令的信息不足、DAgger/PPO 权重及随机相位训练造成的分布缺口、执行控制/接触动力学。训练日志的平均回报和平均回合长度不能替代完整越障测试。下一步应拍摄障碍前缘 2–3 秒的 Teacher/Student 配对视频，记录足端接触与高度，并做 `teacher action` 接管和 oracle terrain/height-scan 消融；在定位前不建议直接重训 30k iteration。

## 复现与文件

- Student 诊断入口：`scripts/php_student/eval_student_full.sh`（设置 `STUDENT_EVAL_DIAGNOSTICS=1`、新的 `STUDENT_EVAL_REPORT_DIR`）。
- Teacher 配对入口：`scripts/php_student/eval_teacher_paired.py`，程序化覆盖保存配置，以绕过当前官方 Tyro 对异构随机化字典的解析错误。
- Student 结果：`reports/php_student/paired_seed42_student5/full_clip_seed42.json`。
- Teacher 结果：`../new_teacherpolicy/reports/php_official/paired_seed42_teacher5/full_clip_seed42.json`。
- 先前未固定障碍 seed 的 Student 100 次/条报告为 0/4000；可作为现象复现，但**不可与旧 Teacher 报告视为同地形配对因果实验**。
