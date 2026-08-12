# Airline/Retail 局部后果方向预实验

这个实验承接“最终成功方向”预实验的负结果，但不重新生成任何 τ²-Bench 轨迹。
它把已有的每条完整离线轨迹拆成真实的局部转换：

```text
当前环境状态 s_t + Agent 已生成的工具动作 a_t
                    ↓ 确定性重放
工具结果 + 真实下一状态 s_{t+1}
```

目标是验证一个更局部的问题：模型生成动作以后、工具结果返回以前，其隐藏表示
里是否已经存在“这一步会把环境推向正确目标”的简单方向。这比用第一步预测整条
轨迹的最终 reward 更接近后续的 `(s,a) -> consequence` 轻量 verifier。

## 标签完全由代码产生

实验复用以下两份完整结果，不调用 Agent 或 User 生成新内容：

```text
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_airline_base/results.json
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_retail_base/results.json
```

对每个任务，代码先按 τ²-Bench 官方 evaluator 的方式，在干净环境中重放任务的
参考动作，得到官方目标数据库。随后按原顺序重放日志中的全部工具调用以恢复
真实状态，但只把代码明确声明为会修改状态的 Agent 调用保留为学习样本。每个
保留样本得到三类标签：

- `tool_success`：落盘工具结果是否成功，并要求确定性重放得到相同结果；
- `state_changed`：重放前后的数据库是否真的有字段变化；
- `goal_progress`：仅对代码中明确声明会修改状态的工具定义。若下一状态与官方
  目标数据库的不同字段数减少，则为正，否则为负。

不使用 LLM 补标签，不把参考动作序列当成 Agent 唯一允许走的流程。参考动作只
用于构造官方最终目标状态；任何不同动作只要使真实数据库更接近同一目标，都可
得到正标签。不存在 user tools 的 Airline/Retail 会把 user state 记为空值，
不会伪造用户反馈。

## 主实验与诊断

每个真实写操作保留同一上下文链的三个隐藏表示时刻：

1. `before`：动作生成前，只能表示模型打算做什么；
2. `action`：动作和参数已经生成，工具结果还没有返回；
3. `result`：真实工具结果已进入上下文。

主实验预先固定为：`action + goal_progress + 第 47 层 + 工具中心化的同工具内
AUROC`。拟合前先用 Train 中该工具的平均表示减去工具中心；测试时也只使用
Train 工具中心，再只把同一个工具的正例和负例互相比较。这样同时防止方向拟合
和评分阶段仅凭工具名拿到高分。这是严格的
`(s,a) -> consequence` 读出：没有看到结果，不存在把答案直接抄进特征的泄漏。
`before` 用于判断动作意图，`result` 是应当更容易的结果可读性检查；第 31 层和
工具成功/状态变化标签都只做诊断。

方向仍采用无超参数的正负中心差。只用官方 Train 任务拟合，在官方 Test 任务上
以同工具内 AUROC 为主指标，同时报告普通 AUROC。一个轨迹可能含多个工具决策，
因此置信区间按轨迹整体重采样，不能把同一轨迹的相关步骤错误当作独立样本。
报告还给出“只看工具名的 Train 成功率”
等基线；若隐藏方向没有超过工具名基线，就不能声称模型内部学到了额外风险信号。

## 两道运行门槛

第一步先运行纯 CPU `audit`，确定性重放全部离线写操作并生成标签支持报告。确认
Train/Test 都有正负例、且至少一个相同工具在两边都可比较后，再运行单决策
smoke。smoke 从 Retail 任务 0 中自动找到一个真实写操作，验证：

- 原始工具结果与重放结果完全一致；
- 三段上下文严格递增且来自同一决策；
- vLLM 导出的 token 与实际提交 prompt 一致；
- 第 31/47 层维度都是 5120、有限且非零。

smoke 通过后再跑完整 8 卡隐藏表示导出。完整阶段首先在 CPU 上重放和审计所有
离线写操作；任何结果不一致都会直接报错，不会静默落下错误标签。

## 产物

```text
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_local_consequence/
├── local_consequences.jsonl
├── local_consequence_support.json
├── activation_requests.jsonl
├── activation_shards/
├── activations.jsonl
├── local_consequence_report.json
└── smoke/
    ├── activation_requests.jsonl
    ├── activations.jsonl
    └── smoke_report.json
```

`local_consequence_support.json` 是完整实验前最重要的人工审核文件：它按领域和
Train/Test 报告可用写操作数量、三个标签的正负支持度，以及哪些工具同时存在
正负例。如果某一领域的 `goal_progress` 在 Train 或 Test 只有一个类别，正式
报告会明确标记不可评测，
不会用别的标签偷偷替代主问题。
