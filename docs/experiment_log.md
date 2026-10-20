# Experiment Log

## Qwen2.5-1.5B release

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

