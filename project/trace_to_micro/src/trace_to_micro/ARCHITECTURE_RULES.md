# Trace-to-Micro 源码文件与模块变更约束

本文档是 `src/trace_to_micro` 的源码组织约束。这里的“必须”“禁止”是合并
代码前需要满足的规则，不是建议。目标是让每个文件都有稳定、可解释的职责，
避免按一次实验、一次报错或一个临时想法不断增加零散模块。

适用范围包括：新增、移动、拆分、合并、重命名和删除本目录中的源码文件。
研究笔记、运行脚本、配置和实验产物不应放进本目录。

## 1. 固定目录与职责边界

源码根目录只允许保留以下入口文件和本约束文档：

- `__init__.py`：包说明或经过审查的稳定公共导出；禁止放实现逻辑。
- `config.py`：配置读取、校验和配置模型。
- `cli.py`：解析命令并分发；禁止实现实验流程、指标或数据转换。
- `ARCHITECTURE_RULES.md`：本约束文档。

其余源码必须放入下表已有的责任目录：

| 目录 | 允许承担的职责 | 禁止承担的职责 |
| --- | --- | --- |
| `data_model/` | 可序列化记录、枚举和纯数据结构 | 文件 I/O、模型调用、实验编排 |
| `utils/` | 无领域决策的 JSONL、消息序列化等通用小工具 | 业务流程、指标、模型提示词、兜底逻辑集合 |
| `replay/` | 确定性的状态重放、状态差分、参考轨迹和分支执行 | 跨任务统计、模型生成、完整实验编排 |
| `analysis/` | 对已有任务、轨迹和结果的只读提取与分析 | 实时模型调用、修改环境状态、实验启动 |
| `evaluation/` | 支持度、配对比较、评分与报告聚合 | 上下文生成、服务管理、轨迹生成 |
| `clean/` | clean context 的确定性证据、Semantic Brief、Action Contract 和门控 | 完整实验循环、跨轨迹指标、服务启动 |
| `runner/` | 调用模型、串联阶段、管理单次实验流程和落盘 | 定义底层数据模型、通用状态算法 |
| `runtime/` | 外部模型服务的存活、就绪和兼容性检查 | 实验指标、业务数据分析、服务部署策略 |

目录之外的内容按以下规则放置：

- 命令行薄包装放在 `project/trace_to_micro/scripts/`。
- TOML 等实验配置放在 `project/trace_to_micro/configs/`。
- 研究计划和分析笔记放在仓库的 `research/`。
- 运行生成的 JSON、JSONL 和日志放在配置指定的输出目录，禁止写进 `src/`。
- Notebook、临时调试文件和一次性数据转换脚本禁止放进 `src/`。

## 2. 依赖方向

依赖总体只能从编排层指向能力层：

```text
config / data_model / utils / replay
                  ↓
analysis / evaluation / clean / runtime
                  ↓
                runner
                  ↓
                  cli
```

具体约束如下：

1. `data_model`、`utils`、`replay`、`analysis`、`evaluation`、`clean` 和
   `runtime` 禁止导入 `runner` 或 `cli`。
2. `cli.py` 只能组合已有 API，不能成为其他模块复用逻辑的来源。
3. `runner` 可以调用底层能力，但底层能力不能通过回调、延迟导入或
   `TYPE_CHECKING` 绕过依赖方向。
4. 新依赖禁止造成 Python 模块级循环导入。基础层内部确需协作时，导入具体
   模块或函数，禁止依赖 `__init__.py` 的隐式重导出。
5. 禁止使用 `from ... import *`。`__init__.py` 不得为解决导入路径问题复制
   实现或注册副作用。
6. 仅为了缩短导入路径而增加转发文件或兼容别名文件，必须有明确的外部公共
   API 兼容需求和删除期限；内部重构不得默认保留这类壳文件。

## 3. 新增文件的准入规则

新增源码文件前，提交者必须能够依次回答以下问题：

1. 它属于上表中的哪个唯一职责目录？
2. 使用 `rg` 搜索后，为什么现有同职责文件不能容纳这段代码？
3. 新文件代表的稳定能力是什么，而不是哪次实验或哪个临时 case？
4. 它的直接调用方、输入输出和对应测试分别是什么？
5. 它新增的导入是否遵守第 2 节，并且不会产生循环依赖？

任一问题没有清楚答案时，禁止新增文件，应先把实现放进职责最接近的现有模块。

允许新增文件的必要条件是：代码形成一个可独立测试、具有单一变化原因的完整
能力。以下情形本身都不足以成为新增文件的理由：

- 只有一个私有辅助函数、常量、提示词或 Pydantic 模型；
- 只是为了让原文件变短；
- 只服务某一次运行、某一个 task id 或某一个 bad case；
- 与现有函数同义，仅参数名、模型名或输出文件名不同；
- 预留未来实现的空模块、只有 TODO 的模块或没有调用方的“框架”；
- 为捕获尚未出现的错误而建立 `fallback`、`compat` 或 `safe` 包装层。

文件超过约 300 行只触发一次职责复查，不自动要求拆分。判断依据是职责和依赖，
不是行数。若确需建立新的子包，还必须同时：

- 证明现有八个子包均无法准确表达该职责；
- 在本文档中增加职责边界和依赖位置；
- 更新 `tests/test_architecture.py` 中的显式目录白名单；
- 增加与新包路径对应的测试目录；
- 禁止创建只有 `__init__.py` 和一个零散函数的“形式化分包”。

## 4. 如何选择现有落点

常见改动应优先放入以下已有模块，而不是重新造近义文件：

| 需求 | 首选落点 |
| --- | --- |
| 新的实验记录字段或序列化数据结构 | `data_model/models.py` |
| JSON/JSONL 读写 | `utils/io.py` |
| 对话消息规范化、token 估计或 transcript 序列化 | `utils/messages.py` |
| 状态规范化与状态差分 | `replay/state.py` |
| 从离线 `(s,a,s')` 编译局部后果标签 | `replay/consequence.py` |
| 参考动作重放 | `replay/reference.py` |
| 在同状态上执行候选动作 | `replay/branching.py` |
| 已落盘模型轨迹的解析 | `analysis/logged_trace.py` 或 `analysis/trajectory.py` |
| 已落盘结果汇总 | `analysis/results.py` |
| task/split 清单 | `analysis/task_inventory.py` |
| transition 支持度 | `evaluation/support.py` |
| 同状态、不同上下文的配对指标 | `evaluation/paired.py` |
| clean context 的证据、语义、契约、门控 | `clean/` 中对应现有模块 |
| 完整轨迹或 paired-probe 的运行流程 | `runner/trajectory.py` 或 `runner/paired_probe.py` |
| 模型实验总编排 | `runner/model_experiment.py` |
| vLLM/OpenAI-compatible endpoint 就绪检查 | `runtime/server_preflight.py` |
| 隐藏表示导出、紧凑编码与临时文件清理 | `runtime/activations.py` |
| 成功方向的中心计算、迁移评测与普通信息基线 | `evaluation/success_direction.py` |
| 局部后果方向与按轨迹聚类的不确定性评测 | `evaluation/local_consequence.py` |
| 成功方向实验的跨领域阶段编排 | `runner/success_direction.py` |
| 局部后果请求、隐藏表示分片和评测编排 | `runner/local_consequence.py` |

隐藏表示实验额外遵守以下约束：

- `analysis/` 只从官方落盘轨迹编译事实请求，不允许伪造模型内部表示。
- `runtime/activations.py` 只能负责模型接口与向量序列化，禁止在其中拟合或挑选
  最优层。
- vLLM 导出的逐词隐藏表示临时文件必须在读取选定层、最后一个位置后立即删除；
  长期产物只保存配置声明的层，并使用紧凑的半精度编码。
- 主层必须在配置中预先声明。逐层结果可以作为诊断报告，但禁止在测试任务上选择
  层或分类阈值后再报告为主结果。
- 成功方向必须在任务级训练/测试划分上拟合和评测；同一条轨迹的多个决策不得跨
  划分，统计时每条轨迹权重相同。
- 局部后果标签必须由已落盘动作的确定性环境重放产生，禁止用 LLM 猜测下一
  状态；主标签只在官方声明为会修改状态的工具上定义。
- 局部后果主实验固定读取已经生成的 Airline/Retail 完整轨迹，不得为增加正负
  样本重复 rollout；同一轨迹只能属于一个任务划分，区间估计必须按轨迹聚类。
- `(s,a)->consequence` 主读出点必须包含已生成动作但不能包含工具结果；动作前和
  结果后表示只能作为诊断，禁止在测试结果出来后替换主读出点。

只有当需求与这些文件的变化原因确实不同，且满足第 3 节时，才新增模块。

## 5. 拆分、移动和解耦规则

拆分按“职责和副作用边界”进行，禁止按行数平均切块。推荐的可拆边界包括：

- 纯解析与模型/环境调用；
- 确定性构造与 LLM 补充；
- 单条样本计算与数据集级聚合；
- 外部服务检查与实验业务流程；
- 可序列化数据模型与执行逻辑。

执行拆分或移动时必须满足：

1. 先保持 CLI 参数、配置键、输出 schema、默认输出路径和计算语义不变。
2. 同一提交中更新 Python 导入、测试 monkeypatch 路径、shell 命令、
   `python -m` 路径和 Markdown 引用。
3. 使用 `rg` 检查旧模块路径没有残留；不能只检查 Python import。
4. 测试目录应镜像新的源码职责，例如 `runner/x.py` 的核心测试放在
   `tests/runner/test_x.py`。
5. 不得在搬文件时顺带改变实验定义或指标。确需一起修改时，必须在提交说明和
   测试中明确区分结构变化与行为变化。
6. 不得用宽泛异常捕获、静默默认值或重复落盘掩盖拆分后的错误；关键失败应直接
   暴露。
7. 兼容壳只允许保护已发布的外部调用路径。项目内部路径应一次性迁移，避免新旧
   两套实现长期并存。

解耦完成的判据不是“文件变多”，而是每个模块能用一句话说明职责，测试可以在
不启动无关外部服务的情况下验证该职责，且依赖方向更清晰。

## 6. 删除和合并规则

删除源码文件前必须完成以下清单：

1. 用 `rg` 搜索文件名、模块路径、导出符号、CLI 字符串和输出 schema 名称。
2. 确认没有 shell 脚本、配置、测试 monkeypatch、文档命令或外部入口依赖它。
3. 若能力仍需保留，在同一提交中迁移所有调用方和测试；禁止先删后补。
4. 删除已经失效的测试，或把仍有效的行为测试迁移到新 owner；禁止仅为测试通过
   而降低断言。
5. 文件删除后若子包只剩无意义的 `__init__.py`，应一并删除子包，并更新本文档
   与架构白名单。
6. 不得因为源码重构删除已有实验轨迹、报告、用户配置或其他用户数据。
7. 合并两个文件时，以职责更准确的文件名为保留名；禁止保留 `new_`、`old_`、
   `v2_` 或日期后缀区分实现。

## 7. 命名与代码形态

- 模块名使用稳定能力名，如 `state.py`、`support.py`、`paired_probe.py`。
- 禁止新增含义模糊的 `common.py`、`helpers.py`、`misc.py`、`manager.py`、
  `temp.py`、`new.py` 或 `v2.py`。若确有通用能力，应能给出更具体名称；否则
  说明职责尚未拆清。
- 禁止把多项不相关功能塞进 `utils`。代码被两个地方调用，不等于它是通用工具。
- 面向实验阶段的公开函数应使用动词表达动作；数据模型使用名词。
- 模块私有实现默认不从 `__init__.py` 重导出。只有稳定、确实供包外使用的 API
  才允许显式导出。
- 测试文件名与源码文件名对应，测试 case 名描述可观察行为，而不是实现步骤。

## 8. 每次结构变更必须验证

至少运行：

```bash
uv run pytest -c project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/tests
uv run ruff check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
uv run ruff format --check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
```

此外必须：

- 对移动或删除的模块执行 `rg '<旧模块名或路径>' project/trace_to_micro`；
- 对 CLI 或脚本入口变化运行相应的 `--help` 或无模型预飞行检查；
- 对输出 schema 变化增加向后读取测试或明确宣布不兼容；
- 对 clean context 变化运行证据边界、门控和最多两次重试的测试；
- 纯组织重构不需要调用远程模型，但不得跳过单元测试和静态检查。

`tests/test_architecture.py` 会机械检查源码根模块、已批准子包和底层到编排层的
部分依赖边界。修改测试白名单不代表自动获得新增目录许可，仍须先满足本文档的
职责与准入规则。

## 9. 提交前检查清单

- [ ] 新文件有唯一 owner，且现有模块确实不适合容纳。
- [ ] 没有临时、空白、重复、兼容壳或按 case 命名的源码文件。
- [ ] 新依赖遵守方向且没有模块级循环导入。
- [ ] 拆分没有暗中改变实验定义、配置、输出或指标。
- [ ] 所有 import、脚本、文档、monkeypatch 和旧路径已经同步。
- [ ] 测试镜像源码职责，关键阶段失败会直接暴露。
- [ ] 研究笔记、配置、运行脚本和生成数据位于 `src/` 之外。
- [ ] pytest、Ruff lint 和 Ruff format check 全部通过。
