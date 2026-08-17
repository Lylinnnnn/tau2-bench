# 冻结 checkpoint 的完整任务评测

这个目录只负责训练结束后的模型推理和官方指标汇总，不参与 GRPO 更新，也不修改
τ²-Bench 的 evaluator。

## 两个阶段

1. `scripts/run_checkpoint_inference_tmux.sh`：从 OSS 中发现所有完整的
   `global_step_*/actor/lora_adapter`，把基座和各 LoRA 平均分配给最多 8 个单卡
   vLLM 服务并行运行。随后让基座及每个 checkpoint 分别完成 Airline、
   Retail 的完整用户对话和环境交互。每个任务仍由原生 `tau2.runner.run_domain`
   执行，因此轨迹中的单任务 `reward_info` 来自官方 evaluator。
2. `scripts/compute_official_metrics.sh`：不调用模型，只读第一阶段严格合并的完整
   轨迹，调用 `tau2.metrics.agent_metrics.compute_metrics` 计算每个领域的
   `pass^1`、其他 `pass^k` 和平均 reward，并给出各 checkpoint 相对基座的差值。

训练日志中的 `val-core/.../reward/mean@1` 是单步期待偏离代理指标，不是这里的
官方 `pass^1`。

## 数据和模型边界

- 默认只运行官方 held-out `test`：Airline 20 个任务、Retail 40 个任务。
- 基座 Agent 与所有 LoRA checkpoint 使用相同任务、用户模型、采样参数、随机种子
  和试验次数；User 始终使用未训练的 Qwen3-32B 基座。
- 默认一条任务运行一次，`pass^1` 就是成功任务比例。最终需要多次试验时设置
  `NUM_TRIALS=4`，但不要把一次和四次试验的结果混到同一个 `RUN_TAG`。
- checkpoint 只从 OSS 读取，不复制或合并成新的全量模型。
- 每个待评测模型分配给一张卡；模型少于 8 个时不会空载多余 GPU，模型多于 8 个时
  按轮转形成均衡队列。同一 LoRA 只加载到负责它的服务，不在 8 个进程中重复占用
  主存。服务使用 HTTP `8200..8207`、互不重叠的内部端口段和独立 RPC 目录。
  为了保证加载的是本次明确列出的 LoRA，这些 HTTP 端口必须事先空闲。
- 中断后使用同一 `RUN_TAG` 重跑会调用 τ²-Bench 的断点续跑，只补缺失任务。

批量查看全部 checkpoint 的 Test 曲线只用于诊断训练是否整体有效，不能在看过曲线后
挑最高点作为“最终模型”，否则等于用 Test 选模型。当前批量脚本会完整报告每一个
checkpoint，不自动选最佳 step。论文主结果应使用训练前约定的最终 step，或在新随机种子
训练中预先固定选择规则。

## 先跑一条任务的全链路检查

训练完成后，先选最终 step，并让每个模型只跑一个 Retail Test 任务：

```bash
cd /home/liuyanlin.lyl/notebook/lyl/tau2-bench
git pull --ff-only origin lyl-dev

CHECKPOINT_ROOT=/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full \
CHECKPOINT_STEPS=50 \
DOMAINS=retail \
MAX_TASKS_PER_DOMAIN=1 \
RUN_TAG=qwen3_32b_full_eval_smoke \
bash project/expectation_step_rl/evaluation/scripts/run_checkpoint_inference_tmux.sh

tmux attach -t expectation-step-rl-infer-qwen3_32b_full_eval_smoke
```

推理成功后单独计算官方指标：

```bash
RUN_TAG=qwen3_32b_full_eval_smoke \
bash project/expectation_step_rl/evaluation/scripts/compute_official_metrics.sh

tmux attach -t expectation-step-rl-metrics-qwen3_32b_full_eval_smoke
```

smoke 的报告会标记 `full_split=false`，只能验证链路，不能作为论文结果。

## 批量评测 OSS 中全部 checkpoint

`CHECKPOINT_STEPS` 留空时，会在脚本启动时自动发现当时已经完整写入 OSS 的所有
step；默认同时运行基座对照。训练恰好正在写入的半成品 step 会被跳过，不影响其他
checkpoint。发现结果是一次启动快照，推理期间新保存的 step 要在下一次运行时评测。
如果显式指定一个不存在或尚未写完的 step，脚本会直接报错。

```bash
CHECKPOINT_ROOT=/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full \
RUN_TAG=qwen3_32b_full_test_t1_s300 \
bash project/expectation_step_rl/evaluation/scripts/run_checkpoint_inference_tmux.sh

tmux attach -t expectation-step-rl-infer-qwen3_32b_full_test_t1_s300
```

完成后汇总：

```bash
RUN_TAG=qwen3_32b_full_test_t1_s300 \
bash project/expectation_step_rl/evaluation/scripts/compute_official_metrics.sh
```

如只评估指定 step：

```bash
CHECKPOINT_STEPS=10,30,50 ...
```

这里的 `test` 是项目用于训练/验证隔离的官方 Test 划分；τ²-Bench 默认对外汇报的
`base` 包含 Train 与 Test。需要生成与默认 `tau2 run` 完全相同任务范围的结果时，新建
不同的 `RUN_TAG` 并显式设置 `TASK_SPLIT=base`，不要覆盖 Test 结果：

```bash
TASK_SPLIT=base \
RUN_TAG=qwen3_32b_full_base_t1_s300 \
bash project/expectation_step_rl/evaluation/scripts/run_checkpoint_inference_tmux.sh
```

## 产物

分片原始轨迹遵循 τ²-Bench 约定写入仓库的 `data/simulations/`。严格合并结果和
报告位于：

```text
project/expectation_step_rl/evaluation/outputs/<RUN_TAG>/
├── trajectories/
│   ├── base/<domain>/<split>/
│   │   ├── results.json
│   │   └── inference_audit.json
│   └── step_<N>/<domain>/<split>/
│       ├── results.json
│       └── inference_audit.json
├── official_metrics.json
├── logs/
└── vllm_logs/
```

`inference_audit.json` 会拒绝任务缺失、重复 trial、缺 reward、基础设施错误以及
Agent/User 模型标识不一致。普通的模型失败（错误工具、超步数、用户终止等）不会
被跳过，而是保留为官方失败样本。
