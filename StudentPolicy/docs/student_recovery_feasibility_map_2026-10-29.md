# Student–Teacher 恢复能力地图（2026-10-29）

## 问题与设置

冻结官方训练得到的 Student `model_29999.pt` 及其 checkpoint 中的 Teacher；clean、40 条 low-climb-76 motion、seed43、frame-0、每条 1 回合。先运行完全无接管基线，并从其配对动作 trace 中提取每条 motion 第一次连续 10 帧 `||a_student-a_teacher||_2 > 1` 的帧。之后分别在该帧前 20 帧、该帧、该帧后 10 帧启动接管。Teacher 仅在标签有效时连续控制 20、40 或 80 步，随后动作权交还 Student；其余配置保持一致。所有分支使用同一 motion/seed，但接管后状态当然不再相同。

这个实验测试的是**有限 Teacher 辅助后，Student 能否继续完成**，不是 Student 自主成绩。它也不是正式的跨 seed 泛化评估。

## 结果

无接管 Student：**0/40**。下表为 Teacher 辅助后的 full-clip 完成数，分母均为 40；数值不能当作 Student 成功率。

| Teacher 开始时刻 | 20 步后交还 | 40 步后交还 | 80 步后交还 |
|---|---:|---:|---:|
| 首次持续动作分歧前 20 帧 | 0 | 11 | 20 |
| 首次持续动作分歧当帧 | 15 | 19 | **27** |
| 首次持续动作分歧后 10 帧 | 4 | 7 | 9 |

晚启动组在 40/80 步配置下平均实际接管约 33/52 步，低于计划值，因为部分回合在窗口结束前已终止；不能将这些回合解释为“Teacher 完成了规定长度的恢复”。基线 39/40 条轨迹先出现动作分歧、后出现 Teacher 标签失效；motion 9 的先后关系相反，不能作为有效的分歧后 Teacher 监督例子。

## 判断

1. 存在明显的可挽救窗口，但它与接管**时机和持续时间**同时相关。更早开始而仅接管 20 步没有作用；晚 10 帧才开始，即便名义接管 80 步也明显较差。因此不能简单把训练窗口整体提前，也不能只加若干单帧标签。
2. 分歧点接管 80 步后有 27/40 完成，说明相当一部分案例在关键时段仍可由 Teacher 纠正并交还 Student；但另 13/40 没有完成，需逐 motion 查看是 Teacher 标签失效、窗口不够、还是交还后再失稳。此前“分歧前 20 帧接管到结束”的 38/40 不等于有限恢复可行率。
3. 本实验不能证明 Student 自己能识别分歧点：触发时刻取自配对 Student–Teacher 动作，属于训练/诊断时可用的 oracle，部署时不能依赖。也不能证明把这些数据加入训练就一定改善自主闭环。

## 更好 checkpoint 的聚焦复核

对提前 15%–55% roll-in 的 `model_30098.pt` 使用**它自己的**无接管 trace 重新定位每条 motion 的分歧点；同样 clean、seed43、40×1。无接管基线为 **3/40**，且 40/40 条均是动作分歧先于标签失效。

| 接管条件 | Teacher 辅助完成 | 相对无接管逐 motion 胜/负/平 |
|---|---:|---:|
| 分歧当帧，20 步后交还 | 9/40 | 8/2/30 |
| 分歧当帧，80 步后交还 | 14/40 | 12/1/27 |
| 分歧后 10 帧，80 步后交还 | 13/40 | 10/0/30 |

该复核支持“关键窗口仍有可救状态”，但**不支持**将原始模型的 27/40 直接外推到新模型，也不支持声称“越晚必然越差”：新模型的后 10 帧分支与当帧分支很接近。接管会使原本成功的个别 motion 失败，说明干预切换本身有代价。每组只有一个 eval seed、每 motion 一回合，结论限于机制筛查。

## 下一步 gate

跨 checkpoint 的聚焦复核已做，显示有可恢复域，但恢复率较原始模型低。下一步先复核每条轨迹的接管开始、标签有效性、终止帧和交还后生存步数；然后做**同 checkpoint、同预算**的短训 A/B：A 保持现有 roll-in，B 加入 Student 自主前缀与 Teacher 连续恢复片段；只对有效 Teacher 标签计入模仿损失，Teacher 控制的 transition 不计入 Student on-policy PPO。最终必须在无接管、多 seed full-clip 上胜过当前 39/480，且不得只提升单一 seed。

代码：`scripts/php_student/eval_student_full.py` 新增 `mapped_window`；入口 `scripts/php_student/run_student_recovery_map.sh`；汇总 `scripts/php_student/summarize_student_recovery_map.py`。结果：`reports/php_student/recovery_map_v1/` 和 `reports/php_student/recovery_map_early15_v1/`。可复核命令：

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_recovery_map.sh summary
STUDENT_RECOVERY_CHECKPOINT=reports/php_student/teacher_rollin_early15_ab_v1/train/teacher/model_30098.pt \
STUDENT_RECOVERY_ROOT=reports/php_student/recovery_map_early15_v1 \
bash scripts/php_student/run_student_recovery_map.sh summary
```
