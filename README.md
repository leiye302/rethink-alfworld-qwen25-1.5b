# RETHINK on ALFWorld — Qwen2.5-1.5B

通过 Route-SFT 与 GRPO + OPSD，让 Actor 自主识别不可靠的 reasoning、触发 `<RETHINK>` 并生成纠正动作；局部 hint 仅在训练时提供给 Teacher。

本仓库是 **Qwen2.5-1.5B-Instruct** 的独立复现入口，沿用 3B 版本的冻结 SFT 样本、配比、损失及评测协议。当前部署为 **8×A100 80GB，基线/方法各四卡并行**；SFT **只训练到 step2000**，但保留旧 4000 步计划的前 2000 步学习率轨迹，接 OPSD 的起点不变。**不是 Qwen3-1.7B**，没有新增 `enable_thinking` 模板开关。1.5B 实验尚未运行，不将 3B 成绩作为本版本成绩。完整差异与验证边界见 [配方审核](docs/experiment_log.md)。

| 实验线 | 初始化 | 后训练 | 启动入口 |
| --- | --- | --- | --- |
| 基线 | Qwen2.5-1.5B-Instruct | GRPO + OPSD，beta=5；无局部 hint/API | `bash scripts/train.sh baseline` |
| 本方法 | 同基模的 Route-SFT step2000 | 相同 GRPO + OPSD；Actor 触发 RETHINK 时额外给 Teacher 局部 hint | `bash scripts/train.sh opsd --allow-api-cost` |

两条实验线共用 `configs/opsd.yaml` 和 `configs/seen140_tasks.json`。除初始化与局部 hint 开关外，训练任务池、采样数量、loss 系数、150 次更新和验证任务/种子保持一致；`prepare.py --baseline` 关闭 hint，并单独保存基线的配置、断点和评测。

## 1. Problem

Agent 可能基于错误的状态理解或计划继续执行动作，而轨迹级奖励难以直接指出哪个 reasoning 步骤需要纠正。我们希望模型在动作执行前自主识别这些问题，并通过训练期的局部指导学习纠错，而不依赖部署时的外部审核器。

## 2. Core Idea

先让 Actor 学会在不可靠 reasoning 后输出 `<RETHINK>`，再在 GRPO + OPSD 中，仅为 Actor 自主触发的步骤向 Teacher 提供局部 hint。

## 3. Why It Matters

- 将轨迹级学习信号与决策级纠错指导结合，关注具体的推理错误。
- 触发由 Actor 自主决定，Analyzer 只提供训练提示，不接管路由。
- 评测时无需 Analyzer 或 hint，保留独立的 Agent 推理与行动流程。

## 4. Method Overview

基模轨迹 → reasoning 标注 → Route-SFT → GRPO + OPSD → ALFWorld 评测

### Route-SFT

从 **Qwen2.5-1.5B-Instruct 基模**直接训练，不经过旧 M1 或额外语义 SFT。使用已经冻结的 **3B 轨迹构造的数据**，不重新采集、标注或替换样本。根据 Actor 实际可见的任务、历史、当前观察及 reasoning 标注 Good/Bad，不确定样本排除。Bad 指当前可见证据支持、尚未自行修正且会误导下一步决策的实质 reasoning 错误。

- **Good**：保留原先自然采样的 action。
- **Bad**：监督 `<RETHINK>` 与经验证的纠正 action；若原 action 已被验证正确，允许保留。
- 路由位置在 reasoning 结束后、action 开始前；草稿 action 不进入路由前缀。

任务、历史及 reasoning 不计算 CE；联合训练 route/action/EOS 后缀，并对路由位置单独计算 marker 二分类损失：

```math
\mathcal{L}_{\mathrm{SFT}} = \mathcal{L}_{\mathrm{suffix\text{-}CE}} + 0.1\,\mathcal{L}_{\mathrm{marker\text{-}BCE}}.
```

后缀 CE 使用 sample-mean；marker BCE 使用全词表中的 `<RETHINK>` 概率，不加 SFT reference KL。训练集为 **10,000 Good + 6,000 Bad**，全局 batch 8、每批 5:3，学习率为 2e-6，运行 **1 epoch / 2,000 steps**。四卡各处理两条，micro batch 每卡仍为 1；5:3 是全局配比，不是每卡配比。开发集为任务互斥的 128 Good + 128 Bad。

**停止步数与 scheduler horizon 分开：** `total_steps=2000`，`scheduler_total_steps=4000`，warmup 仍为 400 步。不能把 cosine 改成在 2000 步衰减到零，否则就不是旧 3B 的 step2000 配方。

### GRPO + OPSD

从 **Route-SFT step2000** 开始，采用 SDAR 的 GRPO + OPSD，门控系数 **beta=5**，训练 **150 updates**；每次更新采集 16 个任务、每任务 8 条轨迹。

整条轨迹参与 GRPO。普通步骤的 Teacher 使用 SDAR 静态 skill；Actor 生成 `<RETHINK>` 的步骤额外获得 API Analyzer 在 rollout 后生成的局部 hint。Analyzer 不决定是否触发；hint 不合格时退回静态-skill Teacher，不改变 Actor 路由。

Student 始终看不到 hint。Teacher 使用当前 Actor 权重的无梯度视角，对 Student 已生成的 token 评分，不替换轨迹中的动作；它与用于 reference KL 的固定参考模型不同。本阶段不使用 episode Analyzer。

## 5. Current Status

代码、冻结数据和运行入口已经准备；本仓库不附带 1.5B 训后模型或性能结果。CPU 数据/分词器检查与单元测试不等同于实际 GPU 训练验收。完整 CUDA 训练、vLLM rollout、hint API 和 ALFWorld 成绩需在运行环境中验证。项目状态见 [project.yaml](project.yaml)。

## 6. Key Results

**Dataset:** ALFWorld `valid_seen`，固定 140 个任务。  
**Metric:** 任务成功率，每个任务一次 rollout。

暂无本版本结果。运行后比较基模直接 GRPO + OPSD 与 Route-SFT step2000 初始化的方法，两者都使用第 150 次更新后的模型、相同任务与种子；评测不使用 Analyzer 或 hint。图表由实际输出生成，仓库没有预填成功率。

## 7. Next Step

- 完成同配方的 1.5B 训练与固定 seen140 评测；本次不改变标签、配比或 loss。

## 8. How to Reproduce

当前部署配置为 Linux/CUDA、**8×A100 80GB**，基线和方法各四卡。并行运行建议至少 44 个可用逻辑 CPU、192 GiB 主机内存（256 GiB 更宽裕），预留 400 GiB 可用磁盘；这是保守部署预算，不是实测最小需求。SFT 在四卡 DDP 上全参训练，OPSD 使用 SDAR 原生四卡 FSDP/rollout，TP=1。冻结 SFT 数据已包含在 [data/frozen_data.zip](data/frozen_data.zip)，无需另找作者索取；还需下载公开基模与 ALFWorld 资源，并配置自己的 hint API。要求见 [data/README.md](data/README.md)。权重、API 密钥、服务器信息和运行日志不进入 Git。程序不依赖作者电脑、外接硬盘或私有服务器。

<details>
<summary>环境与数据准备</summary>

SFT 与 OPSD 使用独立环境：

```bash
git clone https://github.com/leiye302/rethink-alfworld-qwen25-1.5b.git
cd rethink-alfworld-qwen25-1.5b

python3.10 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt

.venv-sft/bin/python src/tools/setup.py --framework
python3.12 -m venv .venv-opsd
.venv-opsd/bin/pip install --upgrade pip setuptools wheel packaging ninja
.venv-opsd/bin/pip install -r configs/requirements-opsd.txt
.venv-opsd/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation
.venv-opsd/bin/pip install -e .deps/sdar -c configs/requirements-opsd.txt
.venv-opsd/bin/pip check

cp .env.example .env
# 编辑自己的 GPU 分组、ALFWorld 路径及 hint API；不要提交填好的 .env。
.venv-sft/bin/python src/tools/setup.py --bundled-data
.venv-sft/bin/python src/tools/setup.py --download-base --accept-model-license
```

基模固定为官方 `Qwen/Qwen2.5-1.5B-Instruct` 的 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 版本；不会自动替换成其他模型。数据文件的 SHA-256、样本顺序和任务拆分与 3B 版本完全相同，分别由 `configs/data.json`、`configs/data_bundle.json` 以及包内 manifest 验证。

按 `.env.example` 修改以下运行设置；相对路径以仓库根目录为准，GPU 编号使用自己获配的卡。请勿提交填好的 `.env`：

```bash
SFT_PYTHON=./.venv-sft/bin/python
OPSD_PYTHON=./.venv-opsd/bin/python
BASELINE_GPUS=0,1,2,3
METHOD_GPUS=4,5,6,7
RAY_ADDRESS=local
ALFWORLD_DATA=/path/to/alfworld
RETHINK_API_ENDPOINT=https://your-provider.example/v1/chat/completions
RETHINK_API_KEY=your_own_api_key
RETHINK_API_CONCURRENCY=4
```

提示 API 须支持 `glm-5.3-flash`、JSON 输出与 `reasoning_effort=low`，使用者自行提供账号；不需要作者的套餐或账户。模型与请求格式固定在 `src/opsd/sdar.patch`，不自动切换模型或供应商。请求包含触发轨迹，使用者应确认其数据允许发给所选供应商。API 没有自动费用上限；`--allow-api-cost` 是明确启动授权，不是预算控制。

ALFWorld 资源通过 OPSD 环境的 `alfworld-download` 工具准备，需包含完整的 `json_2.1.1` 与 `logic`：

```bash
set -a
source .env
set +a
.venv-opsd/bin/alfworld-download
```

固定评测任务与种子见 [configs/seen140_tasks.json](configs/seen140_tasks.json)，资源版本不匹配会停止，而不会换一组任务运行。训练输出路径由 `configs/sft.json` 和启动脚本决定。

</details>

准备完成后，检查配置并运行完整流程：

```bash
bash scripts/reproduce.sh --check
.venv-sft/bin/python -m unittest discover -s src/tools -p 'test_*.py'
PYTHONPATH=src/sft .venv-sft/bin/python -m pytest src/sft/tests -q
.venv-sft/bin/python src/tools/audit_recipe.py --framework
bash scripts/reproduce.sh --run --allow-api-cost
```

完整流程在 **0–3 号卡运行基线**，同时在 **4–7 号卡依次运行 SFT 显存预检 → SFT2000 → 方法 OPSD**；卡号可在 `.env` 调整，组内四卡且两组不能重叠。两条线各自完成后导出 update150 并进行 seen140 最终评测，结果齐全后自动生成 `results/tables/seen140.csv` 与 `results/figures/seen140.png`。没有预填成绩，也不再训练 SFT2001–4000。也可分别运行：

```bash
bash scripts/train.sh sft
bash scripts/train.sh opsd --allow-api-cost
bash scripts/train.sh baseline
EVAL_GPU=0 bash scripts/eval.sh --model /path/to/exported-hf-model --out runs/seen140 --step 150
```

**只跑基线**：准备 OPSD 环境、公开基模和 ALFWorld 游戏资源后，直接执行 `bash scripts/train.sh baseline`，使用 `BASELINE_GPUS` 四卡，不需要先跑 SFT，也不需要 hint API 账号或 `--allow-api-cost`。不要执行 `reproduce.sh --run`，后者会同时运行两条完整实验线。单独评测使用导出的 Hugging Face 模型目录，固定任务集无需作者的外部文件。

SFT 每 100 步记录 dev loss，每 500 步保存权重，500/1000/1500/2000 共四份；第 2000 步保存完整 optimizer/scheduler、四个 rank 的 RNG 和采样位置。OPSD 每 10 步验证、每 20 步保存，保留最近两份完整断点并保存最终 update150。

SFT 默认保留所有已保存权重，不会向作者电脑传输或自动删除它们；空间不足时保留已有断点、报告错误，不能盲目删文件。方法的 SFT/OPSD 必须前一阶段成功退出再接下一阶段；基线可独立并行。首次安装与 GPU 运行前执行 `.venv-sft/bin/python src/tools/check.py --assets`（需在 shell 中导入 `.env`），以及 `.venv-sft/bin/python src/tools/check.py --runtime sft` 和 `.venv-opsd/bin/python src/tools/check.py --runtime opsd`。CPU 检查不替代 CUDA/NCCL/vLLM 的实机检查。

Ray 临时 socket/log 目录使用系统临时目录，位置记录在本次运行的 `runtime_paths.json`；可用 `RETHINK_RUNTIME_ROOT` 指定自己有权限的短路径。训练断点仍保存在 `checkpoints/`。发布时执行 `python src/tools/package.py --out /path/to/new-source.zip`，只打包白名单源码、公开文档与冻结数据包，不要直接压缩整个工作目录；隐私检查同时扫描压缩包内的文本。

评测固定为 seen140、相同任务与种子、最多 50 步、history 2、temperature 0.4、最大生成长度 512；保留特殊 token 以统计 `<RETHINK>`。完整配置见 [configs/eval.json](configs/eval.json)。

代码基于 [SDAR](https://github.com/ZJU-REAL/SDAR)，固定版本见 [configs/upstream.json](configs/upstream.json)。新增代码采用 [Apache-2.0](LICENSE)，保留上游文件许可；[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 权重采用 Apache-2.0，ALFWorld 资源及由其构造的任务内容仍遵循来源许可。模型权重和 ALFWorld 游戏资源不重复上传。

## 9. 8×A100 80GB 服务器：基线与方法独立运行

**一个 clone 即可运行两条线，共用只读基模、数据和依赖，但不共用训练状态。** 分配八张空闲 A100 80GB：四张跑基线，另四张依次跑 Route-SFT 与方法 OPSD。不要在另一 clone 再启动同样的一对任务。现有启动入口已实现四卡，不再需要运行者自行改分布式代码。SFT 要求每卡总显存至少 70 GiB、空闲至少 65 GiB；本配置不适用于 A100 40GB。

| GPU 分组 | 阶段 | 输出 |
| --- | --- | --- |
| `BASELINE_GPUS=0,1,2,3` | 基模 → GRPO+OPSD150 | `checkpoints/baseline/` |
| `METHOD_GPUS=4,5,6,7` | SFT2000 → GRPO+OPSD150＋局部 hint | `checkpoints/route_sft/`、`checkpoints/opsd/` |

SFT 由指定 SFT Python 的 `torch.distributed.run` 启动四个进程，使用独立随机端口；不要在外层再套一次 torchrun。OPSD 由 Ray 调度四个 FSDP/rollout worker，每条线强制独立 `address=local` 集群及临时目录，不连接另一条线的集群。不要对整台服务器执行 `ray stop`。

在调度器或 `tmux` 的持久会话中运行：

```bash
bash scripts/reproduce.sh --run --allow-api-cost
```

配对控制器保存 `runs/pair/state.json` 和各阶段独立日志；只对自己创建的子进程发送退出信号。SFT 预检使用一次真实 forward/backward/Adam step 的一次性模型，不保存为正式权重。只有正式 SFT 成功退出并核验 step2000 权重、续训状态 SHA，才启动方法 OPSD。任何一条线失败，不继续其后续阶段，另一条线可正常完成；不会盲目重启或重复提交。

### 为什么四卡仍是同一 SFT 目标

全局 sampler 先生成原来的八个样本，四个 rank 按位置分配，各两条；没有四个独立 sampler，也不重复采样。每条 loss 乘 `4/8`，DDP 再平均四个 rank 的梯度：

```math
\frac{1}{4}\sum_{r=1}^{4}\sum_{i\in B_r}\frac{4}{8}\nabla L_i
=\frac{1}{8}\sum_{i=1}^{8}\nabla L_i.
```

因此 suffix CE 和独立 BCE 的相对权重不变；梯度裁剪在同步后执行。每 100 步的 256 条 dev、每 500 步的 32 条自由生成也分给四卡执行，按原顺序合并，样本数不乘四。OPSD 的全局 16 任务×8 rollout、PPO mini-batch 256、每卡 micro 32、beta=5 和原生 loss 均不改；框架负责 FSDP 的世界尺寸归一化。多卡的浮点归约、在线采样和负载均衡顺序可能不同，**不声称权重或随机生成结果逐位一致**。

### 中断恢复

先确认旧任务已退出、断点完整、代码/配置未改变：

```bash
bash scripts/reproduce.sh --run --allow-api-cost --resume
```

它跳过本控制器已验收完成的阶段，并从完整 SFT/OPSD 保存状态恢复。若发生在 SFT 第一个 500 步保存前，或某目录只写了一半、没有完整指针，控制器会停下要求检查，不冒充可恢复。已经失败的最终导出/评测目录也需先保留证据并处理，不覆盖或自动删除。

单独恢复命令仍可用，但不能与配对控制器同时运行：

```bash
bash scripts/train.sh sft --resume-latest
bash scripts/train.sh baseline
bash scripts/train.sh opsd --allow-api-cost
```

新四卡配置不能直接使用旧单卡实验的 `setup.json` 或续训状态；请使用干净检出和新输出目录，不强行修改原断点绑定。

### 最终评测与交付

配对入口自动合并两条线各自 update150 的四份 FSDP 分片，并在该线训练退出后用其第一张卡执行现有 TP1 最终评测；在线训练验证仍使用各自四卡。单卡最终评测不改变已固定的模型、任务、种子或生成预算。八卡分组都用于训练，但 API 等待、存盘及实验先后完成时，不保证八卡始终满负载。

评测固定为同一 seen140、同一任务种子、50 步上限、history 2、temperature 0.4、512-token 预算；保留特殊 token，不调用 Analyzer/Teacher。结果分别为：

- `runs/baseline_seen140/output/summary.json`
- `runs/opsd_seen140/output/summary.json`
- `results/tables/seen140.csv`、`results/figures/seen140.png`

如果只单独运行某条线，手动导出/评测：

```bash
set -euo pipefail
set -a
source .env
set +a
ARM=baseline  # 方法改为 opsd
EXPORT="$PWD/checkpoints/${ARM}_hf150"
PYTHONPATH="$PWD/.deps/sdar" "$OPSD_PYTHON" \
  .deps/sdar/scripts/model_merger.py merge --backend fsdp \
  --local_dir "$PWD/checkpoints/$ARM/checkpoints/global_step_150/actor" \
  --target_dir "$EXPORT"
EVAL_GPU=0 bash scripts/eval.sh --model "$EXPORT" --out "runs/${ARM}_seen140" --step 150
```

方法的 `EVAL_GPU` 应改为获配的第一张方法卡，且必须等该卡训练退出。交付代码版本、解析配置、数据/模型 SHA、完整断点、曲线及任务级结果。RETHINK 触发率是含 marker 的**决策步**占比，不是占总 token 的比例。未测指标如实标注；本仓库 CPU/通信测试不等于已完成八卡 A100、NCCL、vLLM 的端到端实测。
