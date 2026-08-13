# 实验 A：执行前的“结果期待”是否存在

## 1. 要验证什么

实验不训练探针，也不事先规定“应该改地址”“支付不应变化”等后果标签。它只问：

> 模型已经决定调用某个工具、但还没看到工具返回时，模型是否已经更期待真实会
> 出现的返回，而不是同一工具在其他真实任务中产生的相似返回？

如果答案为是，后续才值得把“执行前期待—执行后观察的偏离”发展成异常反馈信号。
本实验只是存在性验证，不声称已经得到异常检测器或强化学习奖励。

## 2. 数据怎样构造

数据只来自已经落盘的 Qwen3-32B Airline 和 Retail 完整轨迹，不重新生成任务、
不重复运行同一个问题，也不读取官方目标、参考动作或最终奖励。

当前 Test 轨迹中 Airline 有 87 个、Retail 有 187 个工具决策。经过“执行成功、
单工具调用、至少三个跨任务困难负例”的预注册条件后，离线预飞行实际得到
220 个候选组，覆盖 55 个官方 Test task。每个候选组包含四个真实工具返回：

1. 正例是当前动作在原轨迹中的真实成功返回；
2. 三个困难负例来自同一领域、官方 Test 划分、同一工具、不同任务；
3. 负例按返回的 JSON 字段/类型结构和长度相近程度排序选出；
4. 返回完全相同或来自同一 task 的样本不作为负例。

例如，当前动作是 `get_order_details(order_id=#W1)`，正例是 `#W1` 的真实订单；
负例不是随机报错或机票结果，而是另外三个 Test 任务里
`get_order_details` 的成功订单结果。

## 3. 六个固定对照

每个四候选组都计算下列 3×2 条件：

- `full`：真实完整历史，末尾是当前工具动作；
- `action_only`：保留系统规则、工具说明和当前动作，删除此前对话；
- `shuffled`：用另一个任务的历史替换真实历史，但保留当前动作；
- `raw`：原始文本；
- `identifier_masked`：机械遮盖邮箱、订单号、用户编号、长数字等高基数标识符。

模型分数是工具消息中“返回内容加结束标记”的条件对数似然除以这段新增词元数；
固定的工具角色头不计分。归一化避免短返回天然占优。`shuffled` 检验历史是否
真的提供了对应当前动作的期待，遮盖版本
检验结果是否仅靠复制订单号等标识符。

## 4. 指标与通过标准

主指标是逐 query 的成对排序准确率：真实返回与每个负例逐一比较，胜出记 1，
平局记 0.5。四候选时还报告：

- `Hit@1`：真实返回是否严格排第一；
- `MRR`：真实返回名次的倒数；
- `top margin`：真实返回分数减最高负例分数。

置信区间按官方 task id 重采样，避免同一个任务内多个工具步骤被当成完全独立的
样本。预注册的强证据要求 Airline 和 Retail 都满足：

1. `full:raw` 成对准确率的 95% 区间下界高于随机水平 0.5；
2. `full:raw - shuffled:raw` 的区间下界高于 0；
3. `full:identifier_masked` 点估计仍高于 0.5。

## 5. 运行顺序

先用已经运行的 8000 端口服务做单 query 严格冒烟测试：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_matching.sh smoke
```

冒烟阶段固定 `TENSOR_PARALLEL_SIZE=1`，默认只使用 `SMOKE_GPU=0`。如果目标
端口已经存在健康服务则直接复用，不会重复启动；自动启动时也只启动单卡服务。
它必须生成 24 行分数：1 个 query × 4 个候选 × 3 种上下文 × 2 种文本版本。
Qwen3 在追加工具返回后可能重写动作末尾的模板边界，因此代码通过“真实返回”和
“哨兵返回”两个完整模板的最长共同前缀定位返回内容起点，不要求动作单独模板是
完整模板的严格前缀。若该边界无法定位、vLLM 不返回逐提示词似然、或返回词元与
待评分词元不一致，仍会直接报错，不做近似兜底。

冒烟通过后，使用八个单卡 vLLM 服务并行运行完整实验：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_expectation_matching.sh full
```

第 0 个 worker 默认复用已经运行的 `8000` 端口；其余七个服务使用
`8201..8207`。内部通信端口是八个互不重叠的 1000 端口区间。某个 HTTP 端口
已经有健康的 `qwen3-32b` 服务时，管理脚本会直接复用；否则自动
启动并在 worker 完成后关闭自己启动的服务。服务启动最多等待 1800 秒。

## 6. 需要返回的产物

冒烟审核先推送：

```text
outputs/qwen3_32b_thinking_t06_expectation_matching/smoke/smoke_report.json
outputs/qwen3_32b_thinking_t06_expectation_matching/smoke/expectation_matching_scores.jsonl
outputs/run_logs/expectation_matching/（若报错则提供对应日志）
```

完整实验完成后推送：

```text
expectation_matching_audit.json
expectation_matching_queries.jsonl
expectation_matching_scores.jsonl
expectation_matching_measurements.jsonl
expectation_matching_report.json
outputs/run_logs/expectation_matching/score_shard_*.log
```
