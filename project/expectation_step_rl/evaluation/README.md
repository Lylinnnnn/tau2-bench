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
- checkpoint 始终以 OSS 路径作为来源和实验身份；可选的节点本地暂存只在一次评测
  进程内存在，不会合并或保存新的全量模型。
- 每个待评测模型分配给一张卡；模型少于 8 个时不会空载多余 GPU，模型多于 8 个时
  按轮转形成均衡队列。同一 LoRA 只加载到负责它的服务，不在 8 个进程中重复占用
  主存。服务使用 HTTP `8200..8207`、互不重叠的内部端口段和独立 RPC 目录。
  为了保证加载的是本次明确列出的 LoRA，这些 HTTP 端口必须事先空闲。
- 中断后使用同一 `RUN_TAG` 重跑会调用 τ²-Bench 的断点续跑，只补缺失任务。

## 实验身份和固定基线

每个 `RUN_TAG` 首次启动时写入 `experiment_manifest.json`，固定记录代码版本、训练
run、OSS checkpoint 根目录和 step、模型元数据指纹、τ²-Bench 源码与领域数据版本、
任务范围、采样参数和随机种子。再次使用同一 `RUN_TAG` 时必须与这些内容完全一致，
否则直接拒绝，避免不同实验覆盖到同一目录。

基座结果按完整评测协议生成 `baseline_id`，默认保存在 OSS 的
`/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/evaluation/baselines/<baseline_id>`。
后续实验只有在模型、benchmark、任务和采样协议全部一致时才复用这份结果；否则会
自动生成新的 baseline。`evaluation/outputs/experiment_registry.json` 从各次 manifest
自动重建，集中列出 run、checkpoint step、baseline、状态和官方指标位置。大轨迹、
baseline 和 registry 都是运行产物，不提交 Git。

## 临时暂存到节点本地存储

`/home/liuyanlin.lyl` 当前是 JuiceFS，不是节点本地 SSD。先用 `df -hT /tmp` 或集群
提供的本地盘路径确认文件系统类型和至少约 70 GiB 空间。设置
`STAGE_EVAL_INPUTS=1` 后，脚本会：

1. 在 `LOCAL_STAGE_PARENT` 下创建唯一临时目录；
2. 只复制一份 62 GB 基座模型和本次发现的 LoRA adapter；
3. 让所有评测 vLLM 服务共享这些本地文件；
4. 推理成功、失败或被正常中断时，先关闭服务，再删除整个临时目录。

暂存优先使用 `rsync` 显示整体进度；服务器没有 `rsync` 时自动退回 `cp -a`，模型和
LoRA 使用同一个复制函数，并保持目标目录结构一致，不需要额外安装软件。

脚本默认拒绝把“本地暂存”指向 FUSE。它在复制前计算本次输入大小，并额外要求
5 GiB 空余。`kill -9` 或节点掉电无法触发 shell 清理；这种情况下只需删除名称以
`expectation_step_rl_eval.` 开头的遗留临时目录。

批量查看全部 checkpoint 的 Test 曲线只用于诊断训练是否整体有效，不能在看过曲线后
挑最高点作为“最终模型”，否则等于用 Test 选模型。当前批量脚本会完整报告每一个
checkpoint，不自动选最佳 step。论文主结果应使用训练前约定的最终 step，或在新随机种子
训练中预先固定选择规则。

## 先跑一条任务的全链路检查

训练完成后，先选最终 step，并让每个模型只跑一个 Retail Test 任务：

```bash
cd /home/liuyanlin.lyl/notebook/lyl/tau2-bench
git pull --ff-only origin lyl-dev

CHECKPOINT_ROOT=/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full_ce56a20 \
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

## 临时排除基础设施异常

如果某条 simulation 因 API、服务或 User Simulator 异常被标记为
`infrastructure_error`，严格官方汇总会拒绝不完整矩阵。需要先查看其余样本趋势时，
可以运行独立的 available-case 汇总。它仍调用 τ²-Bench 的 `compute_metrics`，但明确
报告每组指标的原始任务数、实际统计任务数和被排除的 task；不会修改原始轨迹、正式
audit 或 `official_metrics.json`，产物不得作为完整官方结果。

```bash
RUN_TAG=qwen3_32b_full_ce56a20_allsteps_test_t1_s300_v1 \
MODEL_KEYS=base,step_10,step_30 \
DOMAINS=airline,retail \
bash project/expectation_step_rl/evaluation/scripts/compute_available_case_metrics.sh

tmux attach -t \
  expectation-step-rl-available-case-qwen3_32b_full_ce56a20_allsteps_test_t1_s300_v1
```

结果写入该 run 下的 `provisional_available_case_metrics.json`。修复异常并补齐任务后，
仍须运行 `compute_official_metrics.sh` 得到最终指标。

## 批量评测 OSS 中全部 checkpoint

`CHECKPOINT_STEPS` 留空时，会在脚本启动时自动发现当时已经完整写入 OSS 的所有
step；默认同时运行基座对照。训练恰好正在写入的半成品 step 会被跳过，不影响其他
checkpoint。发现结果是一次启动快照，推理期间新保存的 step 要在下一次运行时评测。
如果显式指定一个不存在或尚未写完的 step，脚本会直接报错。
同一 `RUN_TAG` 中断重启时，会自动读取已有 manifest 中的 step 列表，保证断点续跑
不会混入训练期间后来生成的 checkpoint。

```bash
CHECKPOINT_ROOT=/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full_ce56a20 \
STAGE_EVAL_INPUTS=1 \
LOCAL_STAGE_PARENT=/tmp \
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

如果同一正式评测中只有部分 checkpoint 因基础设施错误失败，使用
`EVALUATE_STEPS` 只补跑这些模型；`CHECKPOINT_STEPS` 仍表示 manifest 中冻结的完整
checkpoint 集合，不能缩成失败子集。设置 `EVALUATE_BASELINE=0` 会保留原 baseline
身份但不在补跑进程中启动它。补跑应使用独立 GPU、HTTP、内部通信和 master 端口，
并设置 `WORKER_LOG_SUFFIX` 保留第一次失败日志。τ²-Bench 的 `auto_resume` 会删除已有
`infrastructure_error` simulation，只重新执行失败或缺失任务，正常结果不会重跑。

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
├── experiment_manifest.json
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

/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/evaluation/
└── baselines/<baseline_id>/          # 固定且按协议复用的基座结果

project/expectation_step_rl/evaluation/outputs/
└── experiment_registry.json          # 所有正式评测的派生索引
```

`inference_audit.json` 会拒绝任务缺失、重复 trial、缺 reward、基础设施错误以及
Agent/User 模型标识不一致。普通的模型失败（错误工具、超步数、用户终止等）不会
被跳过，而是保留为官方失败样本。
