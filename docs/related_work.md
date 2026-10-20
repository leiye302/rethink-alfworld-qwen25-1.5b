# Related Work

## Paper 1

**Title:** [What Shared Prefixes Hide: Trajectory Dropout for On-Policy Distillation](https://arxiv.org/abs/2609.33455)

**Venue:** arXiv preprint, 2026

**Key idea:** 指出共享推理前缀可能引起监督衰减（PISA）：Student 过度自信，或 Teacher 与 Student 近乎一致时，有价值的学习信号可能很弱。为此，在训练时限制 Student 对部分推理前缀的访问，同时保留 Teacher 的完整上下文，以增强有效监督。

**Limitation:** 主要从训练时的上下文与梯度机制入手，不直接学习 Agent 在动作执行前判断是否需要纠错的策略。

**Relation to our work:** 两者都关注推理前缀对后续学习信号的影响。与训练期遮蔽前缀不同，我们探索由 Actor 自主生成 RETHINK 建立显式纠错分支，并在触发位置提供局部指导；该工作为分析 action 处的监督强弱提供了机制参考。

---

## Paper 2

**Title:** [OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.26790)

**Venue:** arXiv preprint, 2026

**Key idea:** 从完整的 on-policy 轨迹中提取 episode-level 和 step-level skills，在 Analyzer 识别的关键步骤优先采用局部 skill，其余步骤采用全局 skill，再将指导转化为 token 级自蒸馏优势，与任务奖励共同优化策略。

**Limitation:** 其路由服务于训练期的指导选择，关键步骤由完整轨迹的事后分析确定，而非由 Actor 在动作执行前自主发出纠错信号。

**Relation to our work:** 该工作与我们在局部指导结合强化学习方面直接相关。我们的区别在于将局部 hint 的使用位置与 Actor 在线生成的 RETHINK 绑定，使纠错触发本身成为可学习、并在部署时保留的控制行为，而非仅增加局部 skill。

---

## Paper 3

**Title:** [HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873)

**Venue:** arXiv preprint, 2026

**Key idea:** 利用当前策略回顾完整失败轨迹，定位与失败相关的 action，并生成针对性反馈。Teacher 在反馈条件下评分，Student 不读取反馈，仅在选中的 action 片段上进行自蒸馏，减少无关位置的更新。

**Limitation:** 其监督位置由完整失败轨迹的事后归因决定，未将动作执行前的自主纠错触发作为显式训练目标。

**Relation to our work:** 两者均将局部反馈转化为训练期自蒸馏信号。我们保留整轨迹 GRPO 与常规 OPSD，由 Actor 的 RETHINK 标记选择额外局部 hint 的使用位置；Analyzer 只负责提供 hint，不决定是否触发纠错。
