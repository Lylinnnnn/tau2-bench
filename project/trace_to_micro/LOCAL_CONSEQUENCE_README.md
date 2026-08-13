# Airline/Retail 局部后果方向预实验

这个实验承接“最终成功方向”预实验的负结果，但不重新生成任何 τ²-Bench 轨迹。
它把已有的每条完整离线轨迹拆成真实的局部转换：

```text
当前环境状态 s_t + Agent 已生成的工具动作 a_t
                    ↓ 确定性重放
工具结果 + 真实下一状态 s_{t+1}
```

实验分为两个问题。第一阶段保留原来的诊断：模型生成动作以后、工具结果返回
以前，隐藏表示里是否存在“这一步会让数据库更接近官方最终目标”的简单方向。
第二阶段是当前主实验：隐藏表示能否预测工具执行以后实际会发生什么；结果返回
后，预期后果与真实后果之间的偏离能否暴露明显异常。第二阶段不读取官方目标，
因此不假设一个动作只有一条官方正确路径。

## 标签完全由代码产生

实验复用以下两份完整结果，不调用 Agent 或 User 生成新内容：

```text
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_airline_base/results.json
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_retail_base/results.json
```

对每个任务，代码先在干净环境中重放参考动作里代码明确声明会修改状态的操作，
得到官方用于最终比较的目标数据库。只读参考动作不会改变数据库，因此不参与目标
构造；这也避免旧任务中已经失效的只读查询参数（例如 Retail 任务 2 中不存在的
商品编号）影响目标。如果会修改状态的参考动作失败，代码不会使用残缺目标继续
打标签，而是排除这个任务的全部轨迹，在支持度报告中记录任务号、动作、参数和
原始错误，然后继续审计其他任务。真实轨迹缺少结果、重放不一致或某条轨迹处理时
出现异常，也按整条轨迹隔离并记录异常类型与堆栈，不会中止整批。随后按原顺序
重放日志中的全部工具调用以恢复
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

### 当前主实验：预期后果与偏离反馈

代码把每条真实局部转换自动编译成多组可以同时成立的简单事实：工具是否执行
成功、数据库是否变化、发生了新增/修改/删除、影响的是订单/预订/用户/航班，
地址、商品、行程、付款、状态哪一类字段；还会从真实工具结果自动编译返回值是
对象还是文本、返回值涉及哪类实体和字段，但不保存具体姓名、编号或金额作为
预测标签。例如，取消订单可能同时得到
`execution.success=true`、`effect.changed=true`、`entity.order=true`、
`field.status=true`、`status.cancelled=true`、`output.kind.object=true` 和
`output.entity.order=true`。这些不是 LLM 标签，也不要求动作和官方参考动作一致。

探针固定读取工具动作已经生成、结果还没返回时的第 47 层最后一个 token。它是
一次闭式求解的多标签线性岭回归，不做大模型微调，也不在 Test 上挑参数。为了
确认探针没有只记住工具名或长上下文，报告比较三组输入：

1. `surface`：工具名、决策位置、此前工具错误数、上下文 token 数、参数数量；
2. `hidden`：动作时刻的隐藏表示；
3. `surface_plus_hidden`：两者合并。

主指标先分别平均“执行结果、数据库变化、工具输出”三类后果内部的 AUROC，再
对三类等权平均，最后计算第三组相对第一组的增量。这样不会因为某类标签头更多
而支配总分。区间按 task id 整体重采样计算；只有 Airline 和 Retail 的 95%
区间下界都大于 0，才称为稳定的额外隐藏信号。结果返回后，用每个后果头的预测
值与真实 0/1 结果之间的均方偏差作为 `surprise`。错误检测同时报告全部后果头，
以及去掉 `execution.success` 之后的语义后果头；后一个口径可以检查结果是否真的
违反模型预期，而不是机械利用“执行失败”标签。

### 旧诊断：官方目标进展方向

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
离线写操作。所有任务级和轨迹级问题都会集中记录，成功的轨迹正常产出；单条失败
不会留下半条样本，也不会导致整批退出。只有结果文件无法读取、配置无效或产物
无法写入等无法继续处理任何样本的全局故障才会终止。

## 产物

```text
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_local_consequence/
├── local_consequences.jsonl
├── local_consequence_exclusions.jsonl
├── local_consequence_support.json
├── activation_requests.jsonl
├── activation_shards/
├── activations.jsonl
├── local_consequence_report.json
├── abstract_consequences.jsonl
├── consequence_expectation_predictions.jsonl
├── consequence_expectation_report.json
└── smoke/
    ├── activation_requests.jsonl
    ├── activations.jsonl
    └── smoke_report.json
```

`local_consequence_support.json` 是完整实验前最重要的人工审核文件：
`audit_completed=true` 表示批处理已经扫描完所有可读输入，不表示原始数据没有
问题。它按领域和 Train/Test 报告可用写操作数量、三个标签的正负支持度、排除
数量，以及哪些工具同时存在正负例；`local_consequence_exclusions.jsonl` 则逐项
记录被隔离的问题。如果某一领域的 `goal_progress` 在 Train 或 Test 只有一个
类别，正式报告会明确标记不可评测，不会用别的标签偷偷替代主问题。

已有 `activations.jsonl` 时，第二阶段只在 CPU 上运行，不会启动 vLLM，也不会
重新生成轨迹：

```bash
bash project/trace_to_micro/scripts/run_qwen3_32b_local_consequence.sh expectation
```
