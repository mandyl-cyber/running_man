# Student Roll-in Curriculum 与阶段 Gate

## 目标

以已经复验有效的宽 roll-in checkpoint `wide_rollin_v1/train/model_30198.pt` 为起点，逐步降低 5%–85% phase 内的 Teacher 控制概率，使 Student 从 Teacher 覆盖状态平滑过渡到纯闭环控制。结构、Teacher、深度编码器、窗口、LR、env 数保持不变。

## 阶段

| 阶段 | Teacher 控制概率 | 默认训练预算 |
|---|---:|---:|
| S0 | 1.00（当前 incumbent） | 已完成 |
| S1 | 0.75 | 150 iterations |
| S2 | 0.50 | 150 iterations |
| S3 | 0.25 | 150 iterations |
| S4 | 0.00（Student 闭环、仍用有效 Teacher label） | 150 iterations |

每阶段使用上一个通过 gate 的 checkpoint；优化器重置，LR=1e-4，256 env。每个训练 checkpoint 无论 gate 结果如何都保留。

## Quick gate

每阶段先对 incumbent 和 candidate 做相同的 clean、frame-0、无接管评估：40 motion × 1 episode × seed 43/44，共 80 个配对单元。

`PASS` 必须同时满足：

- candidate 至少多 2 个成功 episode；
- 逐 seed-clip 的胜数大于负数；
- 平均失败进度最多回退 0.02。

成功数下降至少 2，或逐 clip 与失败进度共同明显变差，判定 `REGRESS`；其余判定 `HOLD`。只有 `PASS` 才进入下一阶段。

## HOLD / REGRESS 行为

脚本不会覆盖或删除 checkpoint。发生 `HOLD` 或 `REGRESS` 时立即：

1. 保存 candidate checkpoint、训练 coverage 和 gate JSON；
2. 对 incumbent 与 candidate 分别执行 40 motion × 3 episode × seed 43/44 的完整无接管评估；
3. 写入 `STOPPED_AT` 并停止后续训练。

如果 S1–S4 全部通过，则对最终 checkpoint 做同样的双 seed 完整评估并写入 `FINAL_CHECKPOINT`。

## 命令

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_rollin_curriculum.sh verify
bash scripts/php_student/run_student_rollin_curriculum.sh run
```

所有结果保存在 `reports/php_student/rollin_curriculum_v1`。脚本支持在已完成的评估和阶段 checkpoint 上继续执行，但拒绝复用不完整训练目录。

## S1 HOLD 后的保守修订

原始 S1（p=0.75、150 iterations、LR=1e-4）完整评估由 `30/240` 变为 `32/240`，但平均失败进度下降约 0.024，表现为能力迁移而非一致改善，因此不晋升。保守 profile 改为：

- 概率阶梯：0.90、0.80、0.70、0.60、0.50；
- 每阶段 50 iterations；
- LR=3e-5；
- gate、停止和完整评估规则不变。

```bash
STUDENT_CURRICULUM_PROFILE=conservative \
bash scripts/php_student/run_student_rollin_curriculum.sh verify

STUDENT_CURRICULUM_PROFILE=conservative \
bash scripts/php_student/run_student_rollin_curriculum.sh run
```

保守版输出位于 `reports/php_student/rollin_curriculum_v2_conservative`。

## 保守版执行结果

p090 使用 50 iterations、LR=3e-5 完成后，quick gate 判定 `REGRESS`（15→7 successes，7 胜/15 负）。按协议完成双 seed 完整评估：

- incumbent：seed43 `17/120`，seed44 `13/120`，合计 `30/240 (12.5%)`，平均失败进度 `0.537`；
- p090 candidate：seed43 `15/120`，seed44 `14/120`，合计 `29/240 (12.1%)`，平均失败进度 `0.522`。

p090 checkpoint 已保存在 `stages/p090/train/model_30247.pt`，但不晋升，p080–p050 未启动。两次实验共同表明：将 Teacher 控制概率从 1.0 降到 0.75 或 0.90 都没有产生一致改善；逐 step 随机切换 Teacher/Student 动作可能引入混合控制分布和动作不连续，不应继续沿相同方向降低概率。

当前 incumbent 保持为 `wide_rollin_v1/train/model_30198.pt`。下一项高信息增益实验应在 p=1.0 下用更低 LR 做短继续训练，区分“模型尚未拟合充分”与“Teacher 控制到 Student 闭环的切换机制本身有问题”；如果 p=1.0 继续训练也不改善，应改用 episode/segment 级 roll-in 或显式 recovery 数据，而不是逐 step Bernoulli 切换。
