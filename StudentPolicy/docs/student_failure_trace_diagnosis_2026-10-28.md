# Student 逐步失败诊断（2026-10-28）

## 现状

- 官方式 Student 基线在 clean、frame-0、无接管评估中成功率很低。
- 相同继续训练预算下，单帧本体观测为 `6/120 (5.0%)`，三帧为 `5/120 (4.2%)`；三帧没有改善。
- 本轮固定 seed 43，对 motion 4、12、24、26、38 采集逐步 Student/Teacher 动作、两类观测、reference、机器人状态、接触、深度图、相机位姿与深度延迟。

## 已排除或降级的嫌疑

1. **reference/command 帧错位**：10 条 trace 的 command 与源 motion 逐帧 MAE 均为 0，最佳偏移均为 0。
2. **深度缓存取错帧**：buffer match 最大误差为 0；实际延迟稳定为 3–5 帧，与配置一致。
3. **数值坏图**：深度有限值比例为 100%，图中障碍轮廓可见。
4. **本体单帧导致的主要状态歧义**：三帧 A/B 没有提高总体成功率，逐步失败模式也没有稳定改善。

## 主要证据

- 5 条单帧 trace 的起始 20 步 Teacher–Student action L2 已为 0.315–0.553；三帧为 0.367–0.571。误差不是到障碍接触瞬间才突然产生。
- 单帧组持续 action L2 > 1 的起点约在 frame 155–231，三帧约在 154–242；大多早于 Teacher label 失效/位置终止（frame 217–366）。
- 位置误差持续超过 20 cm 通常出现在 frame 112–297，随后累积至约 1 m 触发失败。
- 可视化深度中能看到障碍，但也存在手臂/身体自遮挡；仅凭图像不能证明编码器提取到了正确的任务几何。
- 既有 Teacher-state 离线探针中，actor-only grouped-CV L2 `0.475`，actor+frozen-depth `0.502`；重新训练 depth encoder 后为 `0.497`，仍未优于 zero-depth `0.482`。这说明当前 depth 表征没有表现出跨 clip 的稳定增益，但离线回归不能直接等同闭环成功率。

## 当前判断

主要瓶颈不是一个已确认的时间戳/路由 bug，也不是单纯缺少三帧历史。更符合证据的是：

> Student 对 Teacher 动作的近似误差从容易阶段已经存在，并在 Student 闭环状态分布下累积；当前局部 Teacher roll-in 只覆盖 25%–55% phase，无法同时修复早期偏移、障碍过渡和后段恢复。深度表征可能是次要放大因素。

## 下一步实验

固定源 checkpoint、单帧结构、优化器重置、学习率、env 数和训练预算，只比较 roll-in 覆盖：

- A：现有 25%–55% Teacher roll-in（已完成，作为 baseline）。
- B：5%–85% Teacher roll-in，使 Teacher 能把早期偏差、主要障碍过渡及后段恢复状态带入训练。

验收必须使用同一 40 clip × 3 episode、seed 43、clean、frame-0、无接管评估。先看总成功率和逐 clip 胜负，再看最早 action divergence 和 failure frame。若 B 不能显著优于 A，则停止继续扩大 roll-in，转向重新训练/重构 depth representation 与 phase-balanced supervised objective。

## 首轮执行结果

B 已按相同源 checkpoint、单帧结构、256 env、200 iterations、fresh optimizer、LR=1e-4 完成：

- A（25%–55%）：`6/120 (5.0%)`。
- B（5%–85%）：`17/120 (14.2%)`。
- 逐 clip：B 相对 A 为 11 胜、3 负、26 平；多个失败 clip 的平均失败进度明显后移。
- B 的各 phase label 有效率约为 95.96%–99.59%，覆盖有效；训练 bundle 内记录了 120 个 clip 路由，评估严格使用 40 个 motion、每个 3 episode、seed 43、无接管。

结论：扩大闭环训练覆盖是当前第一个得到明确正结果的结构化改进，但 14.2% 仍不满足 Student 验收。必须先换 seed 复验；若保持提升，再以 B 为起点逐步收窄 Teacher roll-in，让策略重新适应纯 Student 控制，而不是直接继续扩大网络或历史长度。

seed 44 复验也保持同方向：A 为 `8/120 (6.7%)`，B 为 `13/120 (10.8%)`，逐 clip 为 9 胜、4 负、27 平。两 seed 合并后，A 为 `14/240 (5.8%)`，B 为 `30/240 (12.5%)`，逐 seed-clip 比较为 20 胜、7 负、53 平。因此提升不是 seed 43 的单次偶然结果，但方差仍大、绝对成功率仍低。

## 产物

- `reports/php_student/history_ab_v1_trace/{single,history3}/traces`
- `reports/php_student/history_ab_v1_trace/onset_analysis.json`
- `reports/php_student/history_ab_v1_trace/{single,history3}_depth.png`
- `scripts/php_student/analyze_student_failure_traces.py`
