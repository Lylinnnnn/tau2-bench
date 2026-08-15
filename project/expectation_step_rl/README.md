# Expectation-Step RL

这个项目验证一种比完整 Agent rollout 更窄的训练范式：从 τ²-Bench 的真实离线轨迹中恢复决策状态，让当前策略只生成一个工具动作，执行该动作后，用冻结模型对真实工具返回的“期待偏离”作为连续奖励进行 GRPO。

核心奖励不判断动作是否等于官方参考动作，也不读取任务最终 reward。对动作 `a` 的真实工具返回 `o`，分别计算：

```text
L_full(i)   = log p(o_i | 完整任务上下文, a, o_<i)
L_action(i) = log p(o_i | 系统说明, a, o_<i)
D(i)        = L_action(i) - L_full(i)
```

选取 `D(i)` 最大的 10% 返回 token 求平均，得到 `D_min-k`。如果完整上下文本应让模型更确信这个结果，但实际结果违背期待，`D_min-k` 会变大。再使用官方 Train 中相同“领域 + 工具”的均值和标准差做标准化：

```text
z = (D_min-k - Train均值) / Train标准差
reward = -clip(z, -5, 5)
```

因此，越符合上下文期待的真实返回，奖励越高；偏离越明显，奖励越低。校准直接覆盖离线轨迹中 485 个成功的 Train 工具结果，优先使用“领域 + 工具 + 返回结构”统计；样本少于 5 时依次退回同工具、同返回结构和同领域统计。Airline 和 Retail Test、官方最终状态、参考动作都不参与校准。

## 一次训练样本怎样产生

以 Retail 某条真实轨迹为例：用户已经说明订单号，Agent 此前查到了用户和订单。数据编译器在下一次 Agent 决策之前截断轨迹，保存：

- Agent 当时真实看见的系统说明、用户对话和工具结果；
- 用于恢复数据库状态的完整消息前缀；
- Retail 官方工具 schema；
- 官方 Train/Test 归属和任务编号。

GRPO 对这个相同状态采样多个单步候选。若候选调用 `get_order_details(order_id="#W2378156")`，τ²-Bench 原始工具会在恢复后的数据库上执行它。冻结的 Qwen3-32B 随后给真实返回打期待偏离分。策略更新只作用于候选动作 token；工具返回不会被当成模型应该模仿的文本。

## 代码和框架组织

```text
expectation_step_rl/
├── configs/                 # smoke / pilot / full 参数和 AgentLoop 配置
├── data/                    # 运行后生成的 Train/Test 决策数据（默认不提交）
├── scripts/                 # 环境安装、数据准备和 tmux 启动
├── src/expectation_step_rl/
│   ├── data/                # 离线轨迹 -> 单步状态
│   ├── expectation/         # 匿名化、概率测量、Train 校准奖励
│   ├── tau2_adapter/        # 官方状态恢复和单工具执行
│   └── verl_adapter/        # 自定义 verl AgentLoop
├── tests/
└── third_party/verl/        # 固定提交的 Git submodule
```

`verl` 固定在提交 `bec9ef74768dd201881cd4e54cd0385e87caae27`（release `v0.7.1`）。训练环境固定使用 vLLM 0.11.0、PyTorch 2.8.0、Transformers 4.57.1、FlashAttention 2.8.1 和 FlashInfer 0.3.1。冻结打分服务继续使用已经跑通 prompt logprob 的独立 vLLM 0.25.1 环境；两者不共享 Python 包。

LoRA rollout 按该版本官方要求使用 `vLLM + safetensors load_format`，并开启逐层权重同步以控制峰值显存。项目没有修改 `verl` 的参数更新算法；自定义部分仅是单步环境和奖励来源。

## 服务器准备

从仓库根目录执行：

```bash
git submodule update --init --recursive project/expectation_step_rl/third_party/verl
bash project/expectation_step_rl/scripts/bootstrap_training_env.sh
```

安装脚本会先安装 vLLM/PyTorch，再安装与 PyTorch 2.8 匹配的 FlashAttention 预编译 wheel，不会在服务器上现场编译 `flash-attn`。如果旧环境在 FlashAttention 处失败，直接重新运行同一个安装脚本即可修复；不要另外执行普通的 `pip install flash-attn`，该命令会进入源码隔离构建并可能报构建环境找不到 `torch`。

数据准备默认读取已经完整生成的 Airline/Retail Qwen3-32B 轨迹，并输出：

```text
project/expectation_step_rl/data/decisions_qwen3_32b_t06/train.jsonl
project/expectation_step_rl/data/decisions_qwen3_32b_t06/test.jsonl
project/expectation_step_rl/data/decisions_qwen3_32b_t06/dataset_report.json
project/expectation_step_rl/data/decisions_qwen3_32b_t06/training_calibration_scores.jsonl
project/expectation_step_rl/data/decisions_qwen3_32b_t06/training_calibration.json
```

## 八卡正式训练

正式脚本一次性管理数据准备、Train 校准和训练：

- GPU 0–3 各运行一个冻结 Qwen3-32B 打分服务，HTTP 端口为 8000–8003；
- GPU 4–7 运行一个四卡 FSDP 训练任务，策略 rollout 使用四卡张量并行；
- 每个冻结服务使用不同的 HTTP、vLLM 内部通信、PyTorch master 端口和 RPC 临时目录；
- 如果 8000–8003 任一端口已有相同模型的健康服务会直接复用，脚本只清理由自己启动的进程；
- 每个候选固定路由到一个打分服务，候选之间分散到四个服务；
- DataLoader 不创建额外 worker，避免训练结束时出现 worker 被系统杀死的告警。

从仓库根目录启动：

```bash
git pull --ff-only origin lyl-dev
cd project/expectation_step_rl
bash scripts/run_full_tmux.sh
tmux attach -t expectation-step-rl-full
```

第一次运行会用四个冻结服务计算 485 条 Train 干净结果的校准分；若中途退出，重启后会保留已完成记录并继续缺失部分，只有 485 条全部齐全才会进入训练。正式日志位于 `outputs/run_logs/full.log`，四个冻结服务的独立日志位于 `outputs/run_logs/scorer_pool/`。

## 先跑四卡 smoke

冻结 Qwen3-32B 打分服务应已在 GPU 0、`127.0.0.1:8000` 正常运行。32B 模型的 smoke 默认让 `verl` 使用物理 GPU 1、2、3、4；训练进程看不到 GPU 0，也不会重复启动打分服务：

```bash
cd project/expectation_step_rl
bash scripts/run_smoke_tmux.sh
tmux attach -t expectation-step-rl-smoke
```

smoke 只取一个 Train 状态，在四卡上生成四个候选并更新一步。四个候选既组成同一状态的最小 GRPO 比较组，也使四卡 FSDP 的每张卡获得一个训练样本。rollout 使用四卡张量并行，避免任一训练卡独自承载完整 32B 推理权重。预飞行检查会直接验证数据无 Train/Test 重叠、校准来源、submodule 提交、训练端 vLLM、PyTorch、FlashAttention、FlashInfer 的精确版本和可导入性，以及打分服务模型。

smoke 只验证全链路是否可运行；正式训练直接使用上面的八卡脚本，不额外插入多状态 pilot 阶段。

## 明确不做的事情

- 不训练世界模型，也不预测完整下一状态。
- 不重新模拟整段用户对话。
- 不把唯一参考轨迹当成唯一正确答案。
- 不用 Test 拟合阈值或校准统计。
- 不把工具返回加入策略监督 token。
