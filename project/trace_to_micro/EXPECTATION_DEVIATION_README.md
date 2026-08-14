# 实验 B：受控工具返回异常与期待偏离

## 1. 实验目的

实验 A 已验证：在工具动作生成之后，模型对真实工具返回的平均条件 token 对数概率
通常高于其他任务的同工具返回。实验 B 进一步验证：

1. 保持实体对应关系的一致匿名化能否消除原始编号，同时不让不同候选坍缩成同一个
   `<IDENTIFIER>`；
2. 对真实返回做 1、2、3 处受控错误编辑时，模型分数是否随异常程度增加而下降；
3. 干净返回与异常返回的分数差能否稳定区分异常；
4. 用官方 Train 的干净返回按工具和返回结构校准后，能否得到跨工具可比较的异常分数。

实验读取已有 Airline/Retail 完整轨迹，不产生新 rollout，不使用官方目标、参考动作
或最终奖励构造异常，也不调用 LLM 生成异常。

## 2. 一致匿名化

匿名映射在同一评分组的“任务上文、当前动作、工具定义、全部候选返回”上共同建立。
相同原值映射为相同占位符，不同原值映射为不同占位符。例如：

```text
alice_smith_1234 → <USER_001>
alice_smith_1234 → <USER_001>
bob_jones_9876   → <USER_002>
#W123456         → <HASH_001>
#W654321         → <HASH_002>
```

因此模型不能复制真实编号，但仍可判断“动作中的实体是否与返回中的实体一致”。实验 B
会用该匿名化重新运行实验 A 的四候选比较；如果任意四候选在匿名化后发生文本坍缩，
评分直接报错。

此前发现的问题准确地说是：Retail 有 32 个决策样本，它们各自的一组四候选在旧遮盖
后坍缩成同一文本；并不是每组有 32 个候选。本实验保持每组四候选不变，以便与实验 A
直接比较，只替换匿名映射规则。
离线构造审计会明确记录旧遮盖发生坍缩的决策数，并要求新匿名化在三种上下文下均为
零坍缩，否则在任何模型调用前直接失败。

## 3. 受控异常构造

对每个官方 Test 的干净工具返回，候选异常遵守以下硬约束：

- 被修改字段的原值已经出现在当前动作前的历史或当前动作中；
- 新值来自官方 Train 中同领域、同工具、同字段路径、同 Python 类型的真实返回；
- 新值不能已经出现在当前 Test 历史中；
- 只改字段值，不改工具名、JSON 字段路径、列表位置或值类型；
- 异常相对原始返回的字符长度变化不超过 10%；
- 三级异常累积修改 1、2、3 个不同字段，并保留每次修改的字段路径、原值、替换值、
  donor task 和 donor decision。

严格规则在当前轨迹上构造出 170 个 Test 决策：Airline 63、Retail 107，覆盖 52 个
Test 任务和 14 种工具。每个决策包含严重度 0、1、2、3 四个版本。

## 4. 评分与指标

每个返回复用实验 A 的平均条件 token 对数概率：

$$
S(r;h,a)=\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(r_i\mid h,a,r_{<i})
$$

三种上下文固定为：完整真实上文、仅系统消息与当前动作、其他任务的错误上文。
主要指标包括：

- 干净返回分数高于异常返回的比例，分别报告严重度 1、2、3；
- 一个决策中分数是否严格满足严重度 0 > 1 > 2 > 3；
- 任意两个严重度的分数顺序是否正确；
- 直接用“真实返回分数减候选返回分数”检测异常的 AUROC；
- 用负平均 logprob 检测异常的 AUROC；
- 按官方 Train 干净返回校准后的异常 AUROC；
- 按官方 task id 重采样的 95% 区间。

校准分别对三种上下文进行。优先使用“领域 + 上下文 + 工具 + 返回结构”的 Train
均值和标准差；支持少于 5 条时，显式回退到“领域 + 上下文 + 工具”。两级都不满足
时，该 Test 行不进入校准指标，但仍保留原始分数和组内干净—异常比较。Test 数据
不参与均值、尺度或阈值选择。

## 5. 运行顺序

先运行严格单 GPU smoke：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_deviation.sh smoke
```

smoke 固定 `TENSOR_PARALLEL_SIZE=1`，默认使用 `SMOKE_GPU=0`。端口 8000 已有健康
Qwen3-32B 服务时直接复用，否则自动启动单卡服务。它评分 1 个受控异常查询、1 个
一致匿名化四候选查询和 1 个与受控样本同工具、同返回结构的 Train 干净校准决策，
共 27 行。

smoke 通过后运行完整实验：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_deviation.sh full
```

完整实验默认使用 GPU 0–7 上八个相互独立的单卡 vLLM 服务。第 0 个 worker 使用
8000 端口，其余使用 8301–8307；每个 worker 均固定 `TENSOR_PARALLEL_SIZE=1`，
不进行多 GPU tensor parallel 通信。分片以完整 work group 为恢复单位；半组落盘会
在重跑时自动丢弃并重算，模型调用错误仍会直接显示，已经完整落盘的组不会重复评分。

## 6. 主要产物

smoke 产物：

```text
outputs/qwen3_32b_thinking_t06_expectation_deviation/smoke/smoke_report.json
outputs/qwen3_32b_thinking_t06_expectation_deviation/smoke/expectation_deviation_scores.jsonl
outputs/run_logs/expectation_deviation/smoke.log
```

完整实验产物：

```text
expectation_deviation_audit.json
controlled_anomaly_queries.jsonl
consistent_rematch_queries.jsonl
expectation_deviation_scores.jsonl
expectation_deviation_measurements.jsonl
expectation_deviation_calibration.json
expectation_deviation_report.json
consistent_rematch_report.json
outputs/run_logs/expectation_deviation/full.log
outputs/run_logs/expectation_deviation/score_shard_*.log
```

## 7. Contextual Min-K 概率探针补跑

补跑不重新生成 Agent 轨迹，也不重复一致匿名后的四候选 rematch。它只读取已经构造
好的 170 个 Test 受控异常决策和 335 个 Train 干净校准决策，并为每个返回计算完整
上下文与 action-only 两套逐 token log probability，共需 2,030 次离线前向评分。

对第 `i` 个返回 token 定义：

```text
token_contextual_deviation(i) =
    logp_action_only(i) - logp_full_context(i)
```

主信号 `contextual_min_k_deviation` 固定取偏离最大的 10% token 的平均值。分数越大，
表示返回中最可疑的一小部分 token 越不受正确任务上下文支持。逐 token 数组只在一次
评分调用中暂存；落盘文件只保存以下四个标量：

- `full_mean_surprisal`：完整上下文下的全文平均惊讶度基线；
- `full_min_k_surprisal`：完整上下文下最意外 10% token 的平均惊讶度基线；
- `mean_contextual_deviation`：全文平均上下文似然比基线；
- `contextual_min_k_deviation`：最意外 10% token 的上下文似然比，固定为主信号。

校准只使用官方 Train 的干净工具返回，固定按“领域 + 工具”估计每个标量的均值和
标准差。Test 不参与分组、均值、标准差、10%比例或 3σ 阈值的选择。报告包含：

- 四个信号校准前后的 AUROC 和 Average Precision；
- 主信号按 task id 重采样的 AUROC/AP 95% 区间；
- 干净返回与三级异常的配对方向、严格单调率和任意严重度顺序准确率；
- 主信号按严重度的均值、中位数，以及 3σ 下的误报率、检出率、精确率、召回率、
  特异度和 F1；
- 每个工具的主指标和未达到 Train 最小校准支持度的工具清单。

先用单 GPU smoke 验证逐 token 对齐和落盘字段：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_deviation.sh min-k-smoke
```

smoke 通过后使用八个独立单卡服务补跑、合并并评测：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_deviation.sh min-k-full
```

新增产物：

```text
smoke/contextual_min_k_scores.jsonl
smoke/contextual_min_k_smoke_report.json
contextual_min_k_score_shards/shard_*.jsonl
contextual_min_k_scores.jsonl
contextual_min_k_measurements.jsonl
contextual_min_k_calibration.json
contextual_min_k_report.json
outputs/run_logs/expectation_deviation/min_k_score_shard_*.log
```
