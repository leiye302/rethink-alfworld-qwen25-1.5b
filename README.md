# RETHINK on ALFWorld — Qwen2.5-1.5B

通过 Route-SFT 与 GRPO + OPSD，让 Actor 自主识别不可靠的 reasoning、触发 `<RETHINK>` 并生成纠正动作；局部 hint 仅在训练时提供给 Teacher。

本仓库是 **Qwen2.5-1.5B-Instruct** 的独立复现入口，沿用 3B 版本的冻结 SFT 样本、配比、损失、训练步数与评测协议，只更换 Actor/Teacher/reference 的 backbone 尺寸。**不是 Qwen3-1.7B**，没有新增 `enable_thinking` 模板开关。1.5B 实验尚未运行，不将 3B 成绩作为本版本成绩。

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

后缀 CE 使用 sample-mean；marker BCE 使用全词表中的 `<RETHINK>` 概率，不加 SFT reference KL。训练集为 **10,000 Good + 6,000 Bad**，全局 batch 8、每批 5:3，学习率为 2e-6，共 2 epochs / 4,000 steps。开发集为任务互斥的 128 Good + 128 Bad。

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

当前参考配置为 Linux/CUDA、单张 80 GB GPU、22 个 CPU 核、至少 48 GiB 主机内存，完整流程保守建议预留 400 GiB 磁盘（包含两条实验线与 SFT 断点）。硬件要求沿用 3B 配置，不声称已测试 1.5B 的最小显存。冻结 SFT 数据已包含在 [data/frozen_data.zip](data/frozen_data.zip)，无需另找作者索取；还需下载公开基模与 ALFWorld 资源，并配置自己的 hint API。要求见 [data/README.md](data/README.md)。权重、API 密钥、服务器信息和运行日志不进入 Git。程序不依赖作者电脑、外接硬盘或私有服务器。

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
# 删除 .env 中未使用的 HF_TOKEN / WANDB_API_KEY 示例值，或换成自己的值。
.venv-sft/bin/python src/tools/setup.py --bundled-data
.venv-sft/bin/python src/tools/setup.py --download-base --accept-model-license
```

基模固定为官方 `Qwen/Qwen2.5-1.5B-Instruct` 的 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 版本；不会自动替换成其他模型。数据文件的 SHA-256、样本顺序和任务拆分与 3B 版本完全相同，分别由 `configs/data.json`、`configs/data_bundle.json` 以及包内 manifest 验证。

在 `.env` 末尾补充以下运行设置；相对路径以仓库根目录为准，GPU 编号使用自己获配的卡。请勿提交填好的 `.env`：

```bash
SFT_PYTHON=./.venv-sft/bin/python
OPSD_PYTHON=./.venv-opsd/bin/python
CUDA_VISIBLE_DEVICES=0
ALFWORLD_DATA=/path/to/alfworld
RETHINK_API_ENDPOINT=https://your-provider.example/v1/chat/completions
RETHINK_API_KEY=your_own_api_key
RETHINK_API_CONCURRENCY=4
```

提示 API 须支持 `glm-5.3-flash`、JSON 输出与 `reasoning_effort=low`，使用者自行提供账号；不需要作者的套餐或账户。模型与请求格式固定在 `src/opsd/sdar.patch`，不自动切换模型或供应商。请求包含触发轨迹，使用者应确认其数据允许发给所选供应商。API 没有自动费用上限；`--allow-api-cost` 是明确启动授权，不是预算控制。

ALFWorld 资源可通过 OPSD 环境的 `alfworld-download` 工具准备（先设置 `ALFWORLD_DATA`），需包含完整的 `json_2.1.1` 与 `logic`。固定评测任务与种子见 [configs/seen140_tasks.json](configs/seen140_tasks.json)，资源版本不匹配会停止，而不会换一组任务运行。`.env.example` 中的 `DATA_ROOT` / `CHECKPOINT_ROOT` 保留模板字段；此冻结配方的实际路径由 `configs/sft.json` 和启动脚本决定，不支持仅修改这两个变量来重定位。

</details>

准备完成后，检查配置并运行完整流程：

```bash
bash scripts/reproduce.sh --check
python -m unittest discover -s src/tools -p 'test_*.py'
bash scripts/reproduce.sh --run --allow-api-cost
```

完整流程依次执行 SFT、方法 OPSD、基线及最终 seen140 评测，会使用 GPU 并调用 hint API。完成后根据本次评测自动更新 `results/tables/seen140.csv` 与 `results/figures/seen140.png`，不使用预填成绩。OPSD 起点固定为 SFT step2000，不使用 step4000。也可分别运行：

```bash
bash scripts/train.sh sft
bash scripts/train.sh opsd --allow-api-cost
bash scripts/train.sh baseline
bash scripts/eval.sh --model /path/to/exported-hf-model --out runs/seen140 --step 150
```

**只跑基线**：准备 OPSD 环境、公开基模和 ALFWorld 游戏资源后，直接执行 `bash scripts/train.sh baseline`，不需要先跑 SFT，也不需要 hint API 账号或 `--allow-api-cost`。不要执行 `reproduce.sh --run`，后者会顺序运行两条完整实验线。单独评测使用导出的 Hugging Face 模型目录，固定任务集无需作者的外部文件。

SFT 每 100 步记录 dev loss，每 500 步保存权重，第 2000/4000 步保存可续训状态。OPSD 每 10 步验证、每 20 步保存，保留最近两份完整断点。

SFT 默认保留所有已保存权重，不会向作者电脑传输或自动删除它们；磁盘不足时等待用户释放空间。完整流程要求 SFT/OPSD 各进程依次退出，不能同时占用同一张卡。首次安装与 GPU 运行前请执行 `python src/tools/check.py --assets`（需在 shell 中导入 `.env`），并用相应环境执行 `python src/tools/check.py --runtime sft` / `--runtime opsd`。CPU 检查不替代实际 CUDA 环境的训练验证。

Ray 临时 socket/log 目录使用系统临时目录，位置记录在本次运行的 `runtime_paths.json`；可用 `RETHINK_RUNTIME_ROOT` 指定自己有权限的短路径。训练断点仍保存在 `checkpoints/`。发布时执行 `python src/tools/package.py --out /path/to/new-source.zip`，只打包白名单源码、公开文档与冻结数据包，不要直接压缩整个工作目录；隐私检查同时扫描压缩包内的文本。

评测固定为 seen140、相同任务与种子、最多 50 步、history 2、temperature 0.4、最大生成长度 512；保留特殊 token 以统计 `<RETHINK>`。完整配置见 [configs/eval.json](configs/eval.json)。

代码基于 [SDAR](https://github.com/ZJU-REAL/SDAR)，固定版本见 [configs/upstream.json](configs/upstream.json)。新增代码采用 [Apache-2.0](LICENSE)，保留上游文件许可；[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 权重采用 Apache-2.0，ALFWorld 资源及由其构造的任务内容仍遵循来源许可。模型权重和 ALFWorld 游戏资源不重复上传。

## 9. 8×A100 80GB 服务器：基线与方法独立运行

**现有入口是单卡配方，不是八卡 DDP。** 在八卡服务器上，先分配两张空闲 A100 80GB：一张跑基线，另一张依次跑 Route-SFT 与方法 OPSD。其余卡不必占用。不要直接套 `torchrun --nproc_per_node=8`，也不要因卡多而改变全局 batch、采样量、loss 归一化或更新数。SFT 当前资源检查要求单卡总显存至少 70 GiB、空闲至少 65 GiB；本节不适用于 A100 40GB。

### 两个独立工作目录

在自己有权限的服务器目录中准备两个 clone，并锁定同一 commit：

```bash
git clone https://github.com/leiye302/rethink-alfworld-qwen25-1.5b.git baseline
COMMIT=$(git -C baseline rev-parse HEAD)
git clone https://github.com/leiye302/rethink-alfworld-qwen25-1.5b.git method
git -C method checkout --detach "$COMMIT"
git -C baseline checkout --detach "$COMMIT"
```

两个目录分别完成第 8 节的环境、公开模型与 ALFWorld 资源准备，以及 `bash scripts/reproduce.sh --check`。冻结 SFT 数据已经随仓库上传，方法端执行 `setup.py --bundled-data` 即可校验解包，不需要作者电脑或任何旧服务器。ALFWorld 游戏资源可共用一份只读目录，训练输出、断点、日志和 API 缓存不得共用。

分别编辑各自的 `.env`：基线设置 `CUDA_VISIBLE_DEVICES=0`，方法设置 `CUDA_VISIBLE_DEVICES=1`，编号以实际获配的空闲卡为准。**启动脚本会读取 `.env`；不要只在命令前改 GPU 环境变量而漏改 `.env`。** 两个目录均补充：

```bash
RAY_ADDRESS=local
```

这使每个任务启动独立 Ray 实例，避免意外连接另一实验的现有集群（见 [Ray 初始化文档](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)）。仓库自动生成各自唯一的短临时目录；不要对整台服务器执行 `ray stop` 或清理其他任务。并行运行需为每条线分别预留第 8 节的 CPU、内存与磁盘资源；不足时顺序运行，科学配置不变。

基线无需 hint API。方法端填入自己的兼容 API endpoint/key，密钥只放未提交的 `.env`，不打印或写入命令参数。删除 `.env.example` 中未使用的示例凭据，不能把占位值当作真实凭据。

### 启动两条线

以下代码块分别在对应 clone 的根目录执行。使用调度器或 `tmux` 保持任务独立于 SSH 连接。首次运行先检查 GPU 空闲、资源和依赖；方法端可运行 `bash scripts/train.sh sft --gpu-preflight-only`，验证真实 forward/backward 和优化器显存后再正式启动。该预检不导出训练模型、不替代正式训练。

基线终端：

```bash
set -euo pipefail
mkdir -p runs/launch
bash scripts/train.sh baseline 2>&1 | tee -a runs/launch/baseline.log
```

方法终端：

```bash
set -euo pipefail
mkdir -p runs/launch
bash scripts/train.sh sft 2>&1 | tee -a runs/launch/sft.log
# 只有 SFT 成功退出后才继续；OPSD 入口固定选择 step2000，不是 step4000。
bash scripts/train.sh opsd --allow-api-cost 2>&1 | tee -a runs/launch/opsd.log
```

`pipefail` 必须保留，防止 SFT 失败却继续启动下一阶段。不要在两个目录各执行一次 `reproduce.sh --run`，否则会重复启动两套完整实验。

方法 SFT 保持 10,000 Good / 6,000 Bad、每批 5:3、batch 8、2 epochs / 4,000 steps、后缀 CE + 0.1 marker BCE、不加 SFT KL。两条 OPSD 均为 150 updates、beta=5、每 10 步同一 seen140 验证、每 20 步保存且只保留最近两份完整断点，并保存最终模型。局部 hint 仅由 Actor 的 `<RETHINK>` 触发，Analyzer 不参与路由判断，Student 不见 hint，Teacher 是当前 Actor 的无梯度视角。

### 中断恢复与验收

- SFT 中断：先核对最后完整断点与数据/配置绑定，再执行 `bash scripts/train.sh sft --resume-latest`；也可用 `--resume-checkpoint global_step_2000` 指定完整 epoch 断点，两种参数不要同时使用。不能用全新启动覆盖已有输出。
- OPSD 中断：核对完整保存状态后，在原 clone 重跑原来的 `baseline` 或 `opsd --allow-api-cost` 命令；配置使用 `resume_mode=auto`。日志显示的最新 update 不一定已经保存，恢复应以完整断点为准。
- 首次启动至少核对初始验证及数次真实更新，确认 rollout、奖励、backward、保存链路正常，不能只凭进程存在宣称训练已跑通。当前公开验证为 CPU 测试，不声称已完成 A100 实跑。
- 异常先留存日志、资源与断点证据；不自动改 loss、比例、语义、评测任务或触发阈值。必要的工程调整应记录并在两条对照线保持一致。

### 最终第 150 步导出与评测

等待相应训练退出并释放其 GPU 后，在各自 clone 根目录运行下面命令。基线用 `ARM=baseline`，方法用 `ARM=opsd`：

```bash
set -euo pipefail
set -a
source .env
set +a
ARM=baseline  # 方法目录改成 ARM=opsd
EXPORT="$PWD/checkpoints/${ARM}_hf150"
PYTHONPATH="$PWD/.deps/sdar" "$OPSD_PYTHON" \
  .deps/sdar/scripts/model_merger.py merge --backend fsdp \
  --local_dir "$PWD/checkpoints/$ARM/checkpoints/global_step_150/actor" \
  --target_dir "$EXPORT"
bash scripts/eval.sh --model "$EXPORT" --out "runs/${ARM}_seen140" --step 150
```

评测必须使用同一 `configs/seen140_tasks.json`：140 个 `valid_seen` 任务、同一任务种子、50 步上限、history 2、temperature 0.4、512-token 生成预算，保留特殊 token；不调用 Analyzer/Teacher。每份结果位于 `runs/<arm>_seen140/output/summary.json`。两份齐全后可在方法目录生成比较图：

```bash
.venv-sft/bin/python src/tools/plot.py \
  --baseline ../baseline/runs/baseline_seen140/output/summary.json \
  --method runs/opsd_seen140/output/summary.json
```

交付时保留代码 commit、环境版本、解析后的配置、数据/模型 SHA、完整断点、训练与验证日志及任务级评测结果。报告成功数/140、成功率以及实际测得的 reasoning、协议和 RETHINK 统计；RETHINK 触发率是含 marker 的决策步占比，不是 marker 占总 token 的比例。未计算或未完成的指标应标为未测，不补填成绩。
