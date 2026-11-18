# Student 恢复边界与下一轮受控实验

## 已完成的恢复边界实验

对 seed43 的 40 条 Student 闭环 trace，持续 10 帧 `Student–Teacher action L2 > 1`
被定义为首次稳定动作分歧。所有轨迹都在 frame 198–250 附近出现该事件。

- 38/40 轨迹在 episode 终止前 Teacher 标签始终有效；
- 另外 2 条也是动作分歧先于标签失效；
- 因而标签屏蔽通常不是首次失稳的触发因素。

从每条轨迹的动作分歧前 20 帧开始，让冻结 Teacher 接管到片段结束，得到
`38/40 (95%)` 成功。该结果不计作 Student 成功；它证明绝大多数 Student 访问的
早期偏差状态仍在 Teacher 的可恢复域中。

结论：当前主瓶颈是 Student 没有稳定内化有效恢复动作，而不是 Teacher 没有能力、
深度数值损坏或标签在分歧前已经消失。

## 下一实验：学习率课程单变量 A/B

原始 30k run 在 iteration 2500 前后已把 LR 从 `3e-4` 降至 `1e-5`。B 组从同一
Teacher、bundle、seed=42、2048 env 重新训练，只将 adaptive KL 的
`desired_kl: 0.01 -> 0.03`，其余保持不变。原始 run 的 500/1000/2500 checkpoint
作为 A 组，避免重复训练。

快速 gate 使用 seed43、40 条、每条 1 episode、clean、frame-0、无接管：

- iteration 2500 出现任意成功增益，或失败进度比 A 高至少 `0.03`；
- iteration 1000 失败进度不得回退超过 `0.02`。

未通过则停止，不扩展到 30k；通过后才做多 seed、多 episode 复验并考虑延长训练。

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_lr_probe.sh run
```
