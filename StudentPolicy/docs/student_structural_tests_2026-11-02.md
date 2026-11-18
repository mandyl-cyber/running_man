# Student 结构性诊断：三个测试（2026-11-02）

## 目的与固定条件

目标是区分：多轨迹混训、Student 可观测性、以及偏差后的恢复监督，哪一项更可能限制闭环表现。使用同一 Student `model_30098.pt`、官方 low-climb 40 条 motion/terrain 配对数据、clean 评估；测试脚本为 [`run_student_structural_tests.sh`](../scripts/php_student/run_student_structural_tests.sh)，机器可读汇总在 `reports/php_student/structural_tests_v1/summary.json`。所有训练与评估产物保存在独立目录，未覆盖原 checkpoint。

## 测试 1：单轨迹专训能否突破失败

选择 `forward_walk0.6s_step0.7s`，将其原 motion 与 terrain 文件逐字节复制成单轨迹 bundle。先评估原 Student，再从相同 checkpoint 用原 PHP Student 算法和保存的配置续训 200 次（256 env；仅替换 registry、env 数和保存间隔），最后以同 seed=43、20 次 frame-0 full-clip clean rollout 评估。Teacher 用同 bundle、seed、episode 数作有效性 gate。

| 策略 | 完整轨迹成功 |
|---|---:|
| Teacher | 20/20 |
| Student 专训前 | 0/20 |
| Student 专训后 | 0/20 |

专训 checkpoint 迭代号从 30098 到 30297，Student 和深度网络权重均发生变化；不是脚本空跑。但优化器学习率只有 `1e-5`，且只做短续训，所以结果说明**这项廉价的单轨迹专训没有解决问题**，不能推出“单轨迹也不可能学会”，更不能单独判定多轨迹混训不是原因。两次 Student 评估各 20 次都以 bad tracking 结束，总执行步数分别约 398/400，失败很早。

## 测试 2：同状态同标签的观测对照

从 `continuation_collapse_v1/paired_rollout_v1/source/traces` 的 40 条 Student 自主轨迹里取障碍相关阶段的 685 个有效 Teacher 标签，按整条 clip 留出，比较对同一 Teacher 动作的离线预测误差（越低越好）。

| 输入/模型 | 跨 clip 平均动作 L2 |
|---|---:|
| Actor-only | 1.914 |
| Actor + 保存的深度特征 | 1.941 |
| 特权 Teacher 观测 | 1.083 |

另做 5 折、短时可训练深度编码器对照：zero-depth 1.965、frozen-depth 2.024、trainable-depth 2.014。保存的深度特征及此短时重训均未带来明显收益；特权输入的误差明显更低，提示 reference/terrain 信息或其表征存在缺口。**这不是证明原始深度图信息不足**：样本少、深度帧稀疏、模型与优化预算有限，离线预测误差也不是闭环成功率。

## 测试 3：冻结 Teacher 的有限恢复接管

对 40 条轨迹各取一次 seed=43 clean rollout；从 Student 持续动作分歧前 20 帧开始，使用冻结 Teacher 接管 80 步，然后交还 Student。无接管 3/40 成功，有限接管 13/40 成功；逐 clip 对比为净挽回 10 条、无新增失败。**13/40 都是 Teacher 辅助的成功，不是可部署 Student 的自主成功。**

同批 40 条自主轨迹的 Teacher 标签有效率按帧段为：0–99 帧 3960/3960（100%）、100–249 帧 5710/6000（95.2%）、250 帧后 2641/8163（32.4%）。这表明前段通常有监督，晚期偏离后标签大量失效；`teacher_valid` 仅表示路由/标签可用，不衡量标签质量。

## 当前判断与下一步

这三项共同支持：核心瓶颈不只是 clip 数量或 Teacher 不会任务；Student 早期闭环偏差、观测/表征与特权状态的差距，以及偏差扩大后的恢复监督缺口共同存在。当前证据**不支持立刻增加同一 low-climb skill 的 Teacher 数量**，因为单个 Teacher 在同轨迹 clean 20/20 且整个 40-clip 集合已有较高成功率。

下一步应先做一组真正受控的训练 A/B：从同一个较早、仍有学习率余量的 Student checkpoint 出发，匹配 env、更新步数与学习率，比较单轨迹和 40 轨迹；两组均测同一条训练轨迹与留出轨迹，并记录前 50 帧失败时刻、Teacher 标签有效率、DAgger/PPO 权重和动作误差。若单轨迹仍无法提升，再检查 Student 输入对早期状态/目标的可辨识性及训练信号，而不是继续增 Teacher 或小幅调 loss。

运行入口：

```bash
cd /home/sjtu/running_man/StudentPolicy
bash scripts/php_student/run_student_structural_tests.sh preflight
bash scripts/php_student/run_student_structural_tests.sh all
bash scripts/php_student/run_student_structural_tests.sh summary
```

`all` 会复用已完成的产物；换 checkpoint、数据或超参数时应设置新的 `STUDENT_STRUCTURAL_ROOT`，不可把不同条件写进同一实验目录。需要 CUDA/Isaac Sim 权限运行仿真和训练。更多参数见脚本头部的 `STUDENT_STRUCTURAL_*` 环境变量。
