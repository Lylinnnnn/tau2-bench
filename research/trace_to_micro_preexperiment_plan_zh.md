# Trace-to-Micro：从稀疏长程离线轨迹提取结构化微任务的预实验计划

> 状态：无训练预实验实施版，2026-08-09
> 对应工程：[`project/trace_to_micro/`](../project/trace_to_micro/)
> 相关研究草案：[`evidence_responsive_agents_tau2_research_plan_zh.md`](./evidence_responsive_agents_tau2_research_plan_zh.md)

## 1. 核心问题

本项目研究的不是“对同一个 query 重复 rollout 后选择更好的轨迹”，而是：

> 在每个具体任务只有一条或极少量历史轨迹、轨迹来自旧行为策略、用户任务彼此不同的条件下，能否从长程日志中恢复可跨任务复用的局部状态转移，并将它们重组为干净的结构化微任务数据？

进一步希望验证以下因果链：

\[
\text{长程上下文负担}
\rightarrow
\text{局部决策准确率下降}
\rightarrow
\text{状态化拆分改善局部决策}
\rightarrow
\text{微任务数据质量提高}
\rightarrow
\text{训练后提升未见长程组合任务}
\]

当前阶段只验证前三项的前提与机制，不训练模型。

## 2. 数据条件与术语

### 2.1 目标数据条件

- offline：只读取已经产生的轨迹；
- off-policy：训练/挖掘数据由目标策略之外的 behavior policy 产生；
- heterogeneous：具体任务、状态、persona 和行为策略可以不同；
- one/few trajectory per instance：不依赖同一 task 的多次 rollout；
- observational first：主方法不主动枚举未观察 action 的反事实结果。

同一 query 的多次 rollout 也可能是 off-policy，但不属于本项目希望解决的稀疏日志条件。

### 2.2 当前 τ² Telecom 能验证什么

[`split_tasks.json`](../data/tau2/domains/telecom/split_tasks.json) 当前包含：

| Split | 数量 | 用途 |
|---|---:|---|
| `small` | 20 | 单原子任务和 oracle 子目标模板 |
| `train` | 74 | 已见原子、已见组合的离线挖掘集 |
| `test` | 40 | 未见精确组合的 LOCO 测试集 |
| `base` | 114 | `train + test` |
| `full` | 2285 | 后续扩大实验规模 |

当前 `train` 和 `test`：

- 精确 task ID 重合数为 0；
- 19 个 atomic issues 在两侧全部出现；
- 因此可验证“已见原子、未见组合”的 compositional OOD；
- 不等价于真正未见 atomic skill 的外推。

当前 Telecom tasks 只初始化一个用户 `John Smith / 555-123-2002`，并只有 5 种 ticket 文本。它不能单独支持“真实多用户/entity OOD”结论。该结论需要外部日志或后续数据集。

## 3. 研究假设

### H0：跨任务局部转移具有足够支持

在不重复具体 task 的情况下，train 中仍存在足够多的局部 state-action-effect 模式覆盖 test 中的未见组合。

### H1：长上下文造成局部决策退化

控制底层环境状态和局部目标后，同一模型在完整长历史下的下一步正确率低于结构化短上下文；差距随 context token、turn index 和 issue 数增加而扩大。

这里首先称为 long-context/context-management degradation。只有配对干预排除了状态与目标差异后，才将恢复效应解释为 attention/context burden 的证据。

### H2：结构化状态与显式分解贡献不同

- 去除无关历史、显式当前状态，主要缓解状态跟踪和信息访问问题；
- 再提供局部子目标，额外缓解任务分解和规划问题。

### H3：结构化微任务是更干净的监督单位

从同一批离线长轨迹中提取的 structured micro samples，相比完整轨迹或普通短窗口，具有更低的 action 冲突、error/no-op 和无关 token 比例。

H3 的训练收益在后续阶段验证，本轮只实现可计算的数据质量指标接口。

## 4. 预实验 A：确定性结构预检

### 4.1 目的

在调用 LLM 或生成真实 simulation 之前，先回答：

1. 现有组合 split 是否满足 LOCO；
2. reference actions 能否在当前环境代码中确定性重放；
3. 每个 action 的状态增量能否准确提取；
4. train 中的 reference transition 对 test 未见组合有多大理论覆盖；
5. 相同 canonical action 的 effect 是否跨组合保持一致。

该实验使用 `evaluation_criteria.actions`，因此只是 oracle structural upper bound，不是 off-policy 实证结果。

### 4.2 数据单位

对 reference trajectory 中每个 action 提取：

\[
e=(task, split, actor, action, args, \Delta s, error, mutating\_annotation)
\]

其中 `Δs` 同时覆盖 assistant DB 和 user DB，以保留 dual-control 同步效果。

### 4.3 执行流程

1. 为每个 task 创建新环境；
2. 执行 initialization data/actions；
3. 依次执行 reference actions；
4. 每个 action 前后分别序列化两个 DB；
5. 计算字段路径级状态差分；
6. 规范化 customer/line/phone 等实体值；
7. 输出 JSONL transition records；
8. 仅用 action/effect 构造支持度统计，atomic labels 不参与匹配。

### 4.4 指标

#### Action coverage

test action 在 train 中至少被多少个不同 task 支持：

\[
Coverage_{action}@k
=
\frac{\#\{e_{test}:support_{train}(actor,action,args)\ge k\}}
{\#e_{test}}
\]

#### Exact-effect coverage

进一步要求 action 对应的字段变化及前后值完全一致：

\[
Coverage_{effect}@k
=
\frac{\#\{e_{test}:support_{train}(action,\Delta s)\ge k\}}
{\#e_{test}}
\]

#### Effect consistency

对每个 canonical action 统计 dominant effect rate 和条件熵：

\[
H(\Delta S\mid actor,action,args)
\]

#### Annotation mismatch

- declared mutating 但 reference 执行无状态变化；
- declared non-mutating 但实际产生状态变化。

这些不一定是 bug，但必须显式列出，避免后续重放错误地过滤事件。

### 4.5 输出与 Go/No-Go

输出：

- `task_inventory.json`；
- `oracle_transitions.jsonl`；
- `oracle_support_report.json`。

决策：

| 结果 | 决策 |
|---|---|
| exact-effect coverage 高且一致 | 进入真实日志审计 |
| action coverage 高但 effect 冲突 | 改进局部状态条件，不构建 DAG |
| action/effect coverage 均低 | τ² 当前组合不足以支撑跨实例挖掘 |
| action-only 已决定几乎全部 effect | 单纯 delta prediction 太简单，主实验转向 utility/handoff |

## 5. 预实验 B：单轨迹真实日志支持度审计

### 5.1 数据

- Telecom `train` 74 个 task，每个最多一条 simulation；
- Telecom `test` 40 个未见组合，每个最多一条 simulation；
- 不在 task 内重复采样；
- 记录 behavior model、user model、seed、persona；
- task ID/atomic labels 只在最后做评价分析。

如果日志来自多个 behavior policies，首先在 train/test 中保持分布基本平衡。held-out behavior-policy OOD 作为后续独立实验，避免与 composition OOD 同时变化而无法归因。

### 5.2 关键统计

#### State recurrence

test transition 的 canonical pre-state/action 在其他 task 中是否有支持。

#### Natural action overlap

相似 pre-state 邻域中是否自然出现至少两个 action，且每个 action 来自至少 `k` 个独立 task。

没有 overlap 时，不能从观察日志声称“另一 action 会导致什么”。方法必须 abstain。

#### Transition prediction

输入 test 的当前 canonical state、actor 和已观察 action，预测：

- changed fields；
- new values；
- error/no-op；
- 是否破坏此前已经满足的状态。

比较：

- action-only majority；
- raw-state kNN；
- query/ticket kNN；
- canonical full-state kNN；
- local transition motif；
- shuffled-state negative control。

如果 action-only 已接近满分，不将 next-state delta 作为主贡献，而转向局部 utility、handoff 和 failure/recovery。

## 6. 预实验 C：同状态、不同上下文的配对机制实验

### 6.1 核心设计

从 composite trajectory 中抽取一个确定的局部决策点，重放到完全相同的底层状态。对同一个冻结模型只改变可见上下文：

| Condition | 输入 | 解释 |
|---|---|---|
| `LONG_RAW` | 完整原始历史 | 真实长程条件 |
| `STRUCTURED_STATE` | 原始全局目标、当前结构化状态、最近反馈、工具 schema | 状态清理/信息访问 |
| `CLEAN_SUBTASK` | `STRUCTURED_STATE` + 局部子目标和成功条件 | 分解上限 |

`CLEAN_SUBTASK` 首轮可以用 `small` task/atomic fix 构造 oracle 上限；主方法最终必须自动从日志中恢复子目标，不能依赖 benchmark labels。

### 6.2 评分

对 assistant action：

- actor/tool/arguments 是否合法；
- action 后状态是否产生预期 progress；
- 是否 error/no-op；
- 是否破坏已修复字段。

对 user handoff：

- agent 是否请求正确用户动作；
- user 在限定微交互窗口内是否执行目标 action；
- 是否实现预期状态增量；
- handoff latency。

### 6.3 分析

重点比较：

- `STRUCTURED_STATE - LONG_RAW`：状态清理收益；
- `CLEAN_SUBTASK - STRUCTURED_STATE`：显式分解收益；
- condition 与 context length/turn index/issue count 的交互。

评测阶段的配对上下文干预不会进入训练集，不改变 one-trace-per-task 的训练数据条件。

## 7. 后续训练实验，但本阶段不实施

从同一批唯一源轨迹构造：

| 训练组 | 数据格式 |
|---|---|
| `FULL` | 完整长轨迹 |
| `WINDOW` | 非结构化局部窗口 |
| `STRUCTURED_MICRO` | 当前状态、局部目标、actor、action、effect |
| `SHUFFLED_MICRO` | 打乱 state-action 配对的负对照 |
| `MIXED` | 完整轨迹与结构化微任务混合 |

必须控制相同源轨迹、target action 数、训练 token budget、optimizer steps 和成功/失败比例。所有模型最终回到 untouched 长程 LOCO tasks 中评测。

主结论只有在以下结果出现时才成立：

> 从唯一的长程 off-policy 日志中提取出的 structured micro data，在相同训练预算下，提高了未见长程组合任务的成功率，并减缓局部决策准确率随 turn/context length 的下降。

## 8. 数据泄漏与结论边界

### 8.1 允许作为 evaluation oracle

- task ID 中的 atomic issue；
- `composed_from`；
- reference actions；
- env assertions；
- `small` task 的局部修复模板。

### 8.2 不允许进入主挖掘输入

- test atomic label；
- test gold next action；
- 使用 test action 的真实 delta 选择 test pre-state 特征；
- 从未来成功片段泄露当前应执行的子目标，而不在方法中说明为 hindsight relabeling。

### 8.3 当前不应声称

- 从纯观察数据识别了无假设的 causal effect；
- τ² Telecom 验证了真实多用户 OOD；
- 对训练期完全未见的 atomic skill 能正确外推；
- oracle reference replay 就是 off-policy 结果；
- 小任务准确率更高自动证明是 attention 导致。

## 9. 工程结构

```text
project/trace_to_micro/
├── configs/                 # 可复现实验配置
├── scripts/                 # 薄入口，不放核心逻辑
├── src/trace_to_micro/
│   ├── config.py            # TOML 配置
│   ├── models.py            # transition 数据结构
│   ├── task_inventory.py    # split 与数据条件审计
│   ├── state_diff.py        # 状态快照、规范化和差分
│   ├── replay.py            # reference/logged trajectory 重放
│   ├── evaluation/          # 与提取逻辑分离的评测代码
│   │   └── support.py       # LOCO 支持度与一致性指标
│   ├── io.py                # JSON/JSONL 输出
│   └── cli.py               # 命令行编排
└── tests/                   # 单元和小型集成测试
```

代码原则：

- 核心逻辑放在可测试模块，脚本只做参数解析和编排；
- 不吞异常；reference replay 出错时直接中止并显示具体 task/action；
- 不做大量兼容性和 fallback 分支；
- 每个关键边界均有测试：task parsing、state diff、replay、support metrics；
- 输出中保留 commit/config，后续补充 reproducibility metadata。

## 10. 当前执行顺序

1. 完成任务结构审计；
2. 完成 reference transition replay；
3. 完成 state diff 和实体值规范化；
4. 输出 LOCO oracle support report；
5. 验证关键测试；
6. 导入或生成每 task 一条的真实 simulation；
7. 将同一提取器切换到真实日志；
8. 根据 natural action overlap 决定是否进入 action comparison；
9. 实现 paired context probe；
10. 只有机制成立后才进入训练。
