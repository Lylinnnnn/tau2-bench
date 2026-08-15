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

因此，越符合上下文期待的真实返回，奖励越高；偏离越明显，奖励越低。样本数不少于 5 的工具使用各自的 Train 统计；更少见的工具使用同领域 Train 统计作为后备，并在日志中标成 `domain_fallback`。这避免把“Train 中出现较少但合法的工具”固定判成负例。Airline 和 Retail Test、官方最终状态、参考动作都不参与校准。

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

`verl` 固定在提交 `bec9ef74768dd201881cd4e54cd0385e87caae27`（release `v0.7.1`）。精确源码的 `setup.py` 声明 `vLLM >=0.8.5, <=0.12.0`，其安装脚本固定 `vLLM 0.11.0`，所以训练环境使用 vLLM 0.11.0 和它依赖的 PyTorch 2.8.0。训练端还使用 verl 官方安装脚本对应的 FlashAttention 2.8.1 预编译 wheel 与 FlashInfer 0.3.1。已有的冻结打分服务是独立进程，可以继续使用已经跑通 prompt logprob 的 vLLM 0.25.1；两者不共享 Python 环境。

LoRA rollout 按该版本官方要求使用 `vLLM + safetensors load_format`，并开启逐层权重同步以控制峰值显存。项目没有修改 `verl` 的参数更新算法；自定义部分仅是单步环境和奖励来源。

## 服务器准备

从仓库根目录执行：

```bash
git submodule update --init --recursive project/expectation_step_rl/third_party/verl
bash project/expectation_step_rl/scripts/bootstrap_training_env.sh
bash project/expectation_step_rl/scripts/prepare_dataset.sh
```

安装脚本会先安装 vLLM/PyTorch，再安装与 PyTorch 2.8 匹配的 FlashAttention 预编译 wheel，不会在服务器上现场编译 `flash-attn`。如果旧环境在 FlashAttention 处失败，直接重新运行同一个安装脚本即可修复；不要另外执行普通的 `pip install flash-attn`，该命令会进入源码隔离构建并可能报构建环境找不到 `torch`。

数据准备默认读取已经完整生成的 Airline/Retail Qwen3-32B 轨迹，并输出：

```text
project/expectation_step_rl/data/decisions_qwen3_32b_t06/train.jsonl
project/expectation_step_rl/data/decisions_qwen3_32b_t06/test.jsonl
project/expectation_step_rl/data/decisions_qwen3_32b_t06/dataset_report.json
project/expectation_step_rl/data/decisions_qwen3_32b_t06/training_calibration.json
```

## 先跑单卡 smoke

冻结 Qwen3-32B 打分服务应已在 `127.0.0.1:8000` 正常运行。smoke 训练默认只让 `verl` 使用 GPU 1，不会重复启动该服务：

```bash
cd project/expectation_step_rl
TRAINING_CUDA_VISIBLE_DEVICES=1 bash scripts/run_smoke_tmux.sh
tmux attach -t expectation-step-rl-smoke
```

smoke 只取一个 Train 状态、生成两个候选并更新一步。预飞行检查会直接验证数据无 Train/Test 重叠、校准来源、submodule 提交、训练端 vLLM、PyTorch、FlashAttention、FlashInfer 的精确版本和可导入性，以及打分服务模型。

确认 smoke 的两个候选都有 `expectation_outcome`、连续 reward 和一次参数更新后，再跑 7 卡 pilot；GPU 0 留给冻结打分服务：

```bash
cd project/expectation_step_rl
bash scripts/run_pilot_tmux.sh
tmux attach -t expectation-step-rl-pilot
```

训练输出和日志位于 `outputs/`。不要先跑 full；pilot 首先要验证奖励组内有方差、工具调用率没有塌缩、Train 奖励改善且 Test 期待偏离没有恶化。

## 明确不做的事情

- 不训练世界模型，也不预测完整下一状态。
- 不重新模拟整段用户对话。
- 不把唯一参考轨迹当成唯一正确答案。
- 不用 Test 拟合阈值或校准统计。
- 不把工具返回加入策略监督 token。
