# Student 官方课程审计与下一步

## 已确认事实

- 环境在 warm-up 结束后由 Student 闭环驱动，Teacher 只在 Student 访问的状态上提供动作标签。
- `ppo_coef` 是 PPO 权重：从 0 线性升高，在 iteration 9000 达到 0.9；DAgger 权重相应降到 0.1。
- 自适应 KL 学习率仅在 `ppo_coef > 0.1` 后启用。
- 当前 30k 训练确实执行过该课程：`model_00000` 为 0，`model_5000` 为 0.5，`model_9500` 起为 0.9。
- 当前发布代码还启用了 phase-residual PPO gate、expert-lost 标签掩码；它们是正式复现实验必须显式记录的算法扩展。
- 随机逐步切换 Teacher/Student action 不是论文主训练方法。已有 continuation 仅从 5.8% 提升到 12.5%，继续提高 Teacher 控制比例没有稳定收益，因此停止该路线。

## 当前假设

Student 的低成功率可能来自两类机制：

1. 从训练早期开始就未学会障碍过渡；
2. 早期 DAgger 曾学会，但 PPO 权重上升、学习率降至 `1e-5` 或后期训练造成遗忘。

必须先比较同一次训练的中间 checkpoint，才能区分二者。

## Checkpoint sweep gate

在相同 clean、frame-0、40 motions、seed 43 合同下评估：

`0, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 29999`

- 若中间 checkpoint 明显优于 29999：定位为课程/遗忘问题，下一实验围绕最佳 checkpoint 调整 PPO ramp、LR 和 early stopping。
- 若所有 checkpoint 都低：定位为表示、标签覆盖或训练分布问题，优先修复有效标签覆盖和困难阶段采样，不再做后期 continuation。
- 任何新训练都必须使用 Student 闭环评估；Teacher takeover 只作为诊断，不计成功率。

运行：

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_checkpoint_sweep.sh run
```

汇总输出：`reports/php_student/checkpoint_sweep_v1/summary.json`。

## 实际结果

11 个 checkpoint 在 seed 43、40 条 clean frame-0 full clips 上均为 `0/40`。
`model_25000` 仅仅是平均失败进度最高的 checkpoint，不是出现过成功轨迹的
checkpoint：

- iteration 0：failure progress `0.089`；
- iteration 1000：`0.276`；
- iteration 7500：`0.392`；
- iteration 10000：`0.519`；
- iteration 25000：`0.536`；
- iteration 29999：`0.529`。

因此可以排除“中期已经学会、后期完全遗忘”。Student 持续学会更多前半段，约在
50%--54% clip progress 进入平台，但从未形成完整障碍过渡闭环。

同时，adaptive KL 把学习率在 iteration 2500（`ppo_coef=0.25`）前后降至
`1e-5` 下限，后续约 27,500 iterations 都维持该下限。这不是单独充分的根因，
但它是下一轮受控实验必须验证的关键变量。

下一实验不再从 `model_29999` 做 action takeover continuation。应从同一个 Teacher
重新训练短程 A/B：保持 Student 闭环与相同数据，唯一改变为官方 adaptive LR 对照
受控的 LR floor / 较慢 PPO ramp；gate 首先看 failure progress 是否越过当前约
`0.54` 平台，再做完整成功率评估。
