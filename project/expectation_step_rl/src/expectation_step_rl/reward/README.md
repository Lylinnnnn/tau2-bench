# Reward 模块

本目录把“产生一个信号”和“怎样融合多个信号”分开。

数据流固定为：

```text
策略生成一个工具动作
  -> tau2_adapter 执行一次
  -> CandidateTransition
  -> 多个 RewardProvider 各自计算原始信号
  -> RewardComposer 融合
  -> RewardDecision.reward 交给 GRPO
```

`interface.py` 定义跨模块数据契约，`expectation.py` 实现当前期待偏离信号，
`composer.py` 实现融合策略，`pipeline.py` 只负责依次调用。`verl_adapter`
不能包含奖励公式。

当前正式实验只有 `expectation` 信号，默认设置为：

```text
final_reward = 1.0 * expectation
```

未来如果得到对每个候选都不同的完整任务官方信号，可以有两种基础融合：

```text
加权：final_reward = w1 * expectation + w2 * official
门控：official 未通过 -> 固定拒绝分；通过 -> 使用 expectation
```

无论采用哪一种，GRPO 最终仍需要一个候选级标量；“非标量融合”指先做约束、
门控或排序，再映射到这个最终标量，并不是把两个对象原样交给训练器。

官方奖励 provider 暂未实现。它必须对每个候选从当前状态继续运行至任务终止，
再调用 τ²-Bench 原生 evaluator。以下做法不属于官方奖励：

- 把离线来源轨迹的最终成功分复制给同组所有候选；
- 把候选是否等于唯一参考动作当作成败；
- 把数据库局部变化或“更接近目标”的分数命名为官方成功分。
