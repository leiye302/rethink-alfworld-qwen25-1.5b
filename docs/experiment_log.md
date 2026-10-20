# Experiment Log

## 当前：八卡部署、SFT 到 2000 步

运行方式为两条独立四卡实验：基线直接 GRPO+OPSD150，方法 SFT2000 → GRPO+OPSD150。SFT 不再跑第二个 epoch；这是省略不会被用作 OPSD 初始化的后半程，不是将学习率重新压缩到 2000 步。

### 相对归档 3B 配方的审核

审核依据是已归档的公开 3B 方法源码和配置，不是其他旧 1.7B/M1、post-action、无 BCE 或外置 router 实验。可独立执行：

```bash
.venv-sft/bin/python src/tools/audit_recipe.py --framework --out reports/recipe_audit.json
```

`configs/reference_3b.json` 保存脱敏的原配方快照及核心文件校验值，程序只允许下表列出的差异，遇到其他训练/评测参数差异会报错。

| 项目 | 本版本处理 |
| --- | --- |
| Backbone | Qwen2.5-3B-Instruct → 同家族 Qwen2.5-1.5B-Instruct，固定各自官方 revision |
| 数据/标签/纠正动作 | 原冻结数据逐文件 SHA 一致，不重采、不重标、不更改 Good 原动作或 Bad 目标 |
| 数据筛选代码/提示词 | 保持一致；仅数据构建脚本的仓库根路径、默认输出路径作公开目录重定位 |
| 编码、mask、sampler、suffix CE、BCE | 核心实现逐文件一致，BCE=0.1、无 SFT KL、全局 batch8/Good5/Bad3 不变 |
| SFT 预算 | 原计划两轮4000步的 **前2000步**；停止后即接后训练 |
| SFT scheduler | horizon4000、warmup400；数值测试逐步对照旧版前2000个 LR，不改成 horizon2000 |
| 并行 | 单卡 → 每条线四卡；SFT 两样本/rank、DDP 正确平均，不把全局 batch 乘四 |
| OPSD | 原 SDAR 的 GRPO、OPSD loss、门控、Teacher/reference、hint 协议源码一致；beta5、150updates、所有科学系数不变 |
| 部署/公开化 | 组间 GPU/Ray/日志隔离，四卡 RNG 保存、端口分配、依赖与路径重定位 |
| 评测 | 公布的方法协议仍为固定 seen140，同清单 SHA/种子/预算；不是历史每一次监控运行的逐位复演 |

**准确表述：** 数据及方法配方与归档 3B 版本对齐，backbone 改为 1.5B；另有明确授权的四卡执行及 SFT2000 停止。不能称“所有配置文件字节完全一样”，也不能保证多卡归约、原生负载均衡及随机 rollout 的逐位一致。

### 新增验证

- 所有2000个全局 batch 与旧第一轮相同；每批5:3且16,000条恰好消费一次。
- 旧/新前2000步的学习率逐步相等；没有因少跑一轮改变 warmup/cosine。
- 独立四副本算术对照，以及 Linux 上的真实四进程 Gloo DDP 对照：变长后缀、CE+BCE、累积缩放、同步梯度、Adam 更新和保存/恢复。
- 配对启动 GPU 互斥、固定2000/150交接、基线无 hint API、两臂共用 OPSD 科学配置。
- 发布前数据/核心代码 SHA、补丁、隐私和干净解包检查；Linux CPU 回归由 `.github/workflows/cpu.yml` 执行。

本地 Windows 的 PyTorch 2.8 wheel 不提供可用 Gloo transport，真实四进程通信测试在该平台显式跳过，不能记成通过。Linux CI 与 GPU 显存预检分别覆盖不同层级；均不能代替运行者的 A100/NCCL/vLLM 首次端到端验收。

## 首次单卡发布记录（已被上述入口替代）

**Scope:** 仅将 3B 配方的 backbone 换成同家族 Qwen2.5-1.5B-Instruct，固定官方 revision。没有引入 Qwen3 thinking 模板，没有重新采集轨迹或重建标签。

**Unchanged:**

- 冻结训练数据 10,000 Good + 6,000 Bad；开发集 128 + 128，按任务隔离。
- Good 保留原采样 action；Bad 为 `<RETHINK>` + 经验证纠正 action。
- reasoning/prompt 无 CE，后缀 sample-mean CE + 0.1 全词表 marker BCE，无 SFT KL。
- batch 8、每批 5:3、LR 2e-6、两轮 4,000 步、上下文 8192，禁止截断。
- OPSD 从 SFT step2000 开始；基线从同一 1.5B 基模开始。两者 beta=5、150 updates、相同 seen140 验证任务与种子。
- 当前 Actor 权重的 Teacher 无梯度评分；局部 hint 只加到 Actor 自主触发的步骤；评测不用 Analyzer。
- 原 SDAR 固定提交、补丁、GRPO/OPSD 损失和资源配置不变。

**Packaging changes:** 配置中的模型身份及对应路径、文档和图表标签；附带原冻结数据包及安全解压入口。不携带旧 3B 成绩、服务器地址、个人联系方式或凭据。

**Validation boundary:** 发布前执行 CPU 单元测试、冻结文件哈希检查、原生 tokenizer 全量边界检查和独立目录运行检查；这些不替代正式 CUDA 训练、vLLM 环境 rollout、API 服务或任务成功率评测。

**CPU release checks (2026-10-20):**

- 21 项 SFT 回归测试通过，包括 loss/梯度边界、5:3 两轮覆盖、单个 Qwen2 小模型更新和完整断点恢复。
- 14 项发布/可移植性测试通过，包括临时独立目录解包运行、无网络 hint 缓存测试、Analyzer 不接管 Actor 路由、无旧成绩占位。
- 1.5B 官方 tokenizer 检查全部 16,256 行：边界错位 0，最大长度 1,474，marker ID 151665，H/R 的 CE token 数为 0。
- 冻结审核库重新执行筛选，五个数据文件的字节与原包完全一致；训练/开发任务无交叉。
- 三个 shell 入口的语法和帮助入口通过；固定 SDAR 提交与补丁可在新依赖目录中获取、应用和校验。
- 公开文件及 ZIP/GZIP 内部文本通过隐私扫描；不上传本机环境、下载权重或配置过的 API 密钥。
- 验证环境为隔离的 CPU Python 3.13、PyTorch 2.8.0+cpu、Transformers 4.51.3。README 的 Linux/CUDA 环境仍需运行端实测。

**Result:** 1.5B 正式实验尚未运行，无成绩。运行完成后由 `src/tools/plot.py` 从两条实验线的真实评测汇总生成图表，禁止以其他模型的结果占位。

