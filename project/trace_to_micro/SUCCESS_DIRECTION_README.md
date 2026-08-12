# Airline/Retail 成功方向预实验

这个预实验只回答一个问题：Qwen3-32B 在客服决策附近的内部表示中，是否存在
能够区分 τ²-Bench 最终成功和失败的简单方向。它不修改模型，也不把参考动作当成
唯一正确流程。

## 实验范围

- 使用官方 `airline` 和 `retail` 的完整 `base` 任务：Airline 50 个，Retail
  114 个。
- 使用官方 `train/test` 任务划分拟合和测试。测试任务绝不参与方向或阈值选择。
- Agent 和 User 都使用已经部署的 Qwen3-32B thinking 模式。
- 默认强制覆盖旧结果，避免把先前不完整或配置不同的轨迹混入实验。
- 只取每条轨迹的第一个客服工具决策，避免事后选择“最容易分”的步骤，也避免
  长轨迹因步骤更多获得更大权重。

## 先通过单任务闸门

完整实验之前必须先跑 Retail 官方 `base` 任务 2。这个 smoke 不用于证明
“成功方向存在”，只验证整条数据链没有接错：

- 只生成任务 2 的一条完整真实轨迹，调用官方 evaluator 得到 reward；
- 直接调用 `tau2.metrics.agent_metrics.compute_metrics` 落盘官方 `pass^1`；
- 从轨迹第一个真实客服工具调用构造 `before/action/result` 三个时刻；
- vLLM 导出文件中的 `token_ids` 必须逐个等于本次提交的 prompt token；
- 三段上下文必须是严格递增的同一条链，输入词数也必须严格增加；
- 第 31、47 层必须各有 5120 个有限、非零数值；相邻时刻不能返回完全一样的
  向量。

向量余弦相似度和欧氏距离会写入报告，但 smoke 不为它们人为设置“合理阈值”。
三个时刻来自同一语义邻域，相似度本来就可能很高；这里能严格证明的是请求、
token、层号、维度和时刻链条对应正确。单条数据没有成功与失败两个类别，报告会
明确写 `hypothesis_evaluable=false`，禁止把 smoke 当作论文结论。

## 三个内部表示时刻

对同一个已经发生的客服工具调用，构造三个事实前缀：

1. `before`：客服还没有生成工具调用；聊天模板加入“客服即将回答”的位置。
2. `action`：工具调用已经生成，但工具结果还没有进入上下文。
3. `result`：真实工具结果已经进入上下文，客服准备继续回答。

三者都来自同一条已落盘轨迹。隐藏表示阶段不会重新让用户回复，也不会重新生成
动作。

## 探针和主指标

对成功和失败训练轨迹分别求单位化隐藏表示的中心：

```text
成功方向 = 单位化（成功中心 - 失败中心）
```

测试轨迹在该方向上的投影就是成功分数。主实验预先固定为：

- 时刻：`before`；
- 标签：官方总体 reward 是否为 1；
- 层：第 47 层；
- 单位：轨迹；
- 指标：测试集 AUROC，0.5 表示随机排序。

每个测试结果还报告分层自助采样的 95% 区间和单侧标签置换检验。一个领域只有在
测试 AUROC 大于 0.5 且置换检验 `p < 0.05` 时才算通过；Airline 与 Retail 都
通过，才称为“两个领域上一致存在成功方向”。只通过一个领域会明确标记为部分
证据，不提升为总的正结论。重复采样只用于预先声明的两个主检验；诊断组合不做
显著性筛选，避免又慢又制造多重比较问题。

第 31 层、其他两个时刻、数据库/沟通分项和跨领域迁移都属于诊断结果，不能在
看过测试结果后替换成“主结果”。报告同时包含对话字符数、消息数、决策位置、
历史工具错误数和输入词数等普通信息基线，防止把“长任务更难”误认为内部成功
信号。

## 为什么分成三个阶段

普通 vLLM 生成服务不会通过标准 OpenAI 接口返回中间层隐藏表示。vLLM 0.25.1
的隐藏表示导出模式可以同时生成并导出指定层，但一个实例只能使用一种服务配置。
因此运行分为：

1. `trajectories`：复用普通 Qwen3-32B 服务，生成完整官方轨迹；
2. `activations`：把轨迹编译为三个事实时刻，再使用隐藏表示导出服务；
3. `evaluate`：只读紧凑隐藏向量并计算报告，不占用 GPU。

隐藏表示服务会暂时写出逐词向量。代码读取第 31、47 层最后一个位置后立即删除
临时文件和锁文件，长期只保存 base64 编码的半精度向量，避免占满服务器磁盘。
读取时遵循 vLLM 的文件锁，确认异步写入完成后才解析；隐藏服务也关闭 CUDA
Graph，优先保证导出向量正确。
服务器支持时，临时文件默认放在内存文件系统 `/dev/shm`；并关闭与该导出功能
不兼容的分块预填充。隐藏服务还会显式使用 vLLM 0.25.1 中支持该功能的兼容
执行器；这不会改变普通轨迹服务。

## 服务复用规则

- `trajectories` 检测到 8000 端口是健康 Qwen3-32B 服务时直接复用，不重复启动。
- `activations` 检测到 8000 已经是健康的隐藏表示服务时直接复用。
- 如果 `activations` 发现 8000 是普通生成服务，会明确退出并要求先停止普通服务；
  它不会杀掉用户已经启动的进程，也不会假装拿到了隐藏表示。
- 只有端口空闲时，脚本才自动启动并在阶段结束后关闭自己创建的服务。

## 产物

完整轨迹：

```text
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_airline_base/results.json
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_retail_base/results.json
```

正式轨迹阶段还会写：

```text
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_success_direction/official_metrics.json
```

其中 `pass^1` 使用仓库官方实现。在每个任务只有一次试验时，它就是成功任务数
除以任务总数；不是我们另外定义的指标。

分析产物：

```text
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_success_direction/
├── trajectory_completeness.json
├── activation_request_coverage.json
├── activation_requests.jsonl
├── activations.jsonl
└── success_direction_report.json
```

`activation_request_coverage.json` 会明确报告没有客服工具决策、因此不能进入
探针的轨迹数量和成功/失败分布。`activation_requests.jsonl` 含对话前缀，不能
当作隐藏表示；真正向量只在 `activations.jsonl` 中。隐藏向量只有在上下文、模型
和层号指纹都一致时才会断点续跑；轨迹阶段默认强制覆盖。

## 运行阶段

### 八卡完整实验（推荐）

服务器有 8 张大显存 GPU 时，使用专用入口让每张卡运行一个单卡
Qwen3-32B 服务。默认对应 GPU `0..7` 和端口 `8100..8107`：

```text
GPU 0 -> 127.0.0.1:8100 -> shard 0
GPU 1 -> 127.0.0.1:8101 -> shard 1
...
GPU 7 -> 127.0.0.1:8107 -> shard 7
```

Airline 和 Retail 的 164 个官方 `base` 任务先合成一个固定列表，再按
位置轮流分配给 8 个工作进程。每个进程写独立的轨迹文件，全部成功后
才严格合并；缺任务、重复任务、模型不一致或基础设施错误都会直接中止。

隐藏状态也按 8 个独立文件提取；同一工具决策的
`before/action/result` 三个时刻一定由同一个工作进程处理。合并时会
检查请求指纹，不允许遗漏、重复或混入旧向量。为避免把未知服务错当成实验
模型，八卡入口要求目标端口在启动前全部空闲，不复用已有服务。

三个阶段可分别运行：

```bash
project/trace_to_micro/scripts/run_qwen3_32b_success_direction_8gpu.sh trajectories
project/trace_to_micro/scripts/run_qwen3_32b_success_direction_8gpu.sh activations
project/trace_to_micro/scripts/run_qwen3_32b_success_direction_8gpu.sh evaluate
```

也可使用 `all` 串行执行三个阶段。“串行”只指阶段顺序；每个 GPU
阶段内仍然是 8 个工作进程并行。启动器仅关闭自己创建的 vLLM 进程。

不要在 8 张卡上已有其他模型服务时运行该入口。端口可用 `BASE_PORT`
修改，GPU 起始编号可用 `FIRST_GPU` 修改。轨迹默认强制覆盖分片旧数据；
要从合法检查点继续时，显式设置 `FORCE_OVERWRITE=0`。

### 单服务运行方式

以下命令都从仓库根目录执行。先创建日志目录并记录当前仓库绝对路径：

```bash
mkdir -p project/trace_to_micro/outputs/run_logs
repo_dir="$(pwd)"
```

先运行单任务 smoke。第一阶段会复用目前健康的普通 Qwen3-32B 服务：

```bash
tmux new-session -d -s tau2_success_smoke_traj \
  "bash -lc 'set -o pipefail; cd \"${repo_dir}\" && project/trace_to_micro/scripts/run_qwen3_32b_success_direction_smoke.sh trajectories 2>&1 | tee project/trace_to_micro/outputs/run_logs/smoke_trajectories.log'"
```

确认该阶段结束后，停止普通服务，再运行隐藏表示阶段：

```bash
tmux send-keys -t qwen3-32b-server C-c
tmux new-session -d -s tau2_success_smoke_hidden \
  "bash -lc 'set -o pipefail; cd \"${repo_dir}\" && project/trace_to_micro/scripts/run_qwen3_32b_success_direction_smoke.sh activations 2>&1 | tee project/trace_to_micro/outputs/run_logs/smoke_activations.log'"
```

第二阶段只有在全部严格检查通过后才会产生：

```text
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_success_direction/smoke/
├── trajectory_completeness.json
├── official_metrics.json
├── activation_requests.jsonl
├── activations.jsonl
└── smoke_report.json
```

`activations.jsonl` 是两个指定层的实际半精度隐藏向量；`smoke_report.json` 是
便于人工审核的维度、范数、距离、token 数和官方分数摘要。中途失败会保留已经
成功落盘的前序数据，但不会生成 `smoke_report.json`，因此不能误判为闸门通过。

第一阶段强制覆盖 Airline 和 Retail 旧轨迹。当前 8000 端口的健康 Qwen3-32B
服务会被自动复用，不会重复启动：

```bash
tmux new-session -d -s tau2_success_trajectories \
  "bash -lc 'set -o pipefail; cd \"${repo_dir}\" && project/trace_to_micro/scripts/run_qwen3_32b_success_direction.sh trajectories 2>&1 | tee project/trace_to_micro/outputs/run_logs/trajectories.log'"
```

轨迹完成后，需要释放普通服务占用的 8000 端口。若当前服务会话名就是
`qwen3-32b-server`，发送中断：

```bash
tmux send-keys -t qwen3-32b-server C-c
```

然后启动隐藏表示阶段。脚本会自行启动专用服务；如果该专用服务已经健康运行，
也会直接复用：

```bash
tmux new-session -d -s tau2_success_activations \
  "bash -lc 'set -o pipefail; cd \"${repo_dir}\" && project/trace_to_micro/scripts/run_qwen3_32b_success_direction.sh activations 2>&1 | tee project/trace_to_micro/outputs/run_logs/activations.log'"
```

隐藏表示提取完成后再启动 CPU 评测：

```bash
tmux new-session -d -s tau2_success_evaluate \
  "bash -lc 'set -o pipefail; cd \"${repo_dir}\" && project/trace_to_micro/scripts/run_qwen3_32b_success_direction.sh evaluate 2>&1 | tee project/trace_to_micro/outputs/run_logs/evaluate.log'"
```

不要在普通生成服务仍占用 8000 端口时启动 `activations`。普通服务完成完整轨迹后
可以停止；隐藏表示阶段不会生成新的对话轨迹。
