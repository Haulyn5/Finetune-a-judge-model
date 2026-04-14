# Finetune-a-judge-model: WildGuardMix 内容安全裁判模型

这个项目从零搭建一个可执行的 LLM 后训练 MVP，目标是基于 `allenai/wildguardmix` 训练一个输出 `label + reason + evidence` 的内容安全裁判模型。

## 目录结构

```text
Finetune-a-judge-model/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
├── requirements.txt
├── README.md
└── scripts/
```

## 0. 环境准备

### 使用 uv 管理环境

推荐 Python 3.10+，并在项目内使用 `uv` 作为默认工作流。

```bash
# 仓库根目录即项目目录
cp .env.example .env
set -a
source .env
set +a
uv venv
source .venv/bin/activate
uv sync
```

如果你还没有安装 `uv`，可先参考官方安装方式安装后再执行以上命令。

### 中国大陆网络环境建议

本项目当前配置与以下说明保持一致：
- `pyproject.toml`：配置 `tool.uv.index-url` 与 `extra-index-url`
- `uv.lock`：锁定当前 `uv sync` 解析结果，便于环境复现
- `.env.example`：提供 `HF_ENDPOINT`、`HF_HOME`、`PIP_INDEX_URL`、`UV_INDEX_URL`
- Hugging Face 缓存默认落在项目内 `.hf-cache`

如需手动导出：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.hf-cache
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

如果仍需使用 pip，也建议：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

脚本不会硬编码代理或镜像地址，只读取你当前 shell 的环境变量。

## 1. 模型策略 / 学习路线

本项目有两条实验线，请明确区分：

1. `03`：**轻量 baseline 对照实验**
   - 使用 DistilBERT 这类小模型快速验证：
     - 数据 schema 是否正确
     - 指标链路是否跑通
     - `safe/unsafe` 二分类是否有一个基础参照值
   - 它只是 reference baseline，**不是**最终结构化裁判模型。

2. `04 -> 08`：**Qwen 4B 主训练路径**
   - 通过 teacher 伪标注、过滤、SFT、评估，训练真正输出
     `label/reason/evidence` 的结构化安全裁判模型。
   - 默认设计继续对齐 **Qwen 4B 级模型**，并假设实验环境可使用 **A100 80GB**。

如果你是第一次读这个项目，推荐先跑通 `01 -> 03` 理解数据与指标，再进入 `04 -> 08` 主路径。

## 2. 一键跑通顺序

按下面顺序运行：

```bash
uv run python scripts/01_inspect_dataset.py
uv run python scripts/02_build_binary_dataset.py
uv run python scripts/03_train_baseline_cls.py
uv run python scripts/04_generate_rationale_pseudo.py
uv run python scripts/05_filter_pseudo_labels.py
uv run python scripts/06_build_sft_jsonl.py
uv run python scripts/07_train_sft_lora.py
uv run python scripts/08_eval.py
```

---

## 3. 各步骤说明

### Step 01: 查看原始数据集结构

```bash
uv run python scripts/01_inspect_dataset.py
```

输出内容：
- dataset splits
- 每个 split 的 features
- 样本预览

用途：先确认 `wildguardmix` 实际字段，避免后续 schema 假设错误。

---

### Step 02: 构建二分类标准数据集

```bash
uv run python scripts/02_build_binary_dataset.py
```

输出文件：
- `data/processed/train.jsonl`
- `data/processed/dev.jsonl`
- `data/processed/test.jsonl`

统一 schema：

```json
{"id":"...","question":"...","response":"...","label":"safe|unsafe","split":"train|dev|test"}
```

默认逻辑：
- 使用 `prompt` 作为 `question`
- 使用 `response` 作为 `response`
- 使用 `response_harm_label` 映射为：
  - `harmful -> unsafe`
  - `unharmful -> safe`
- 自动过滤缺失值与异常标签
- 从训练集切出开发集

这是后续所有脚本共享的数据接口层，所以建议先人工抽查几行输出再继续。

---

### Step 03: 训练轻量 baseline 分类器（对照实验，不是最终主模型）

这一步只用于快速建立 `label` 分类对照指标，方便和后续 Qwen 主模型比较；它不是最终要交付的结构化裁判模型。

```bash
uv run python scripts/03_train_baseline_cls.py \
  --model_name_or_path distilbert-base-uncased \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8
```

输入：
- `data/processed/{train,dev,test}.jsonl`
- 每条样本包含 `id/question/response/label`
- 模型实际看到的是 `question + response` 拼接文本

输出：
- checkpoint: `outputs/baseline_cls/`
- 指标文件: `outputs/baseline_cls/metrics.json`
- 测试集预测: `outputs/baseline_cls/test_predictions.jsonl`

关键指标：
- `accuracy`
- `macro_f1`
- `unsafe_precision`
- `unsafe_recall`
- `unsafe_f1`

为什么看这些指标：
- `macro_f1`：避免只看 accuracy，忽略类别不平衡
- `unsafe_precision/recall/f1`：单独观察高风险类别表现

---

### Step 04: 生成理由+证据伪标注

```bash
uv run python scripts/04_generate_rationale_pseudo.py \
  --teacher_model Qwen/Qwen3.5-27B \
  --input_path data/processed/train.jsonl \
  --max_samples 3000 \
  --sampling_strategy balanced
```

输出：
- `data/interim/pseudo_raw.jsonl`

teacher 目标输出 JSON：

```json
{
  "label": "safe",
  "reason": "...",
  "evidence": ["..."]
}
```

说明：
- 这一步是主路径的开始，不再只是分类，而是开始构造结构化监督信号。
- 当前默认做法是对训练集进行平衡抽样，优先生成 safe/unsafe 更均衡的 teacher 数据。
- Qwen3.5 teacher 通过 `tokenizer.apply_chat_template(..., enable_thinking=False)` 关闭 thinking。
- 脚本会同时保留原始 teacher 文本和解析后的 JSON，便于后续调试解析失败问题。

---

### Step 05: 过滤伪标注

```bash
uv run python scripts/05_filter_pseudo_labels.py
```

过滤规则：
- teacher label 必须与金标签一致
- `evidence` 至少 1 条
- 每条 evidence 必须能在原始 `question` 或 `response` 中匹配
- `reason` 长度合理

输出：
- `data/processed/pseudo_filtered.jsonl`
- `data/processed/pseudo_label_mismatch.jsonl`
- `data/interim/pseudo_dropped.jsonl`

为什么要做这一步：
- 结构化 SFT 对噪声更敏感
- 错误 evidence / 空 reason 会显著拉低训练质量
- 与其追求样本更多，不如先保证监督信号更干净

### Step 05 → Step 04 定向重跑 workflow

当你调整了 teacher prompt，想只对会被 step 05 丢掉的样本重跑时，可以直接使用 dropped 子集：

```bash
uv run python scripts/05_filter_pseudo_labels.py \
  --dropped_output_path data/interim/pseudo_dropped.jsonl

uv run python scripts/04_generate_rationale_pseudo.py \
  --input_path data/interim/pseudo_dropped.jsonl \
  --output_path data/interim/pseudo_rerun.jsonl \
  --sampling_strategy first_n \
  --max_samples 1039 \
  --rerun_tag drop_subset_rerun_v1
```

说明：
- `pseudo_dropped.jsonl` 会保留原始 step 04 行字段，并新增 `drop_reason`
- step 04 会忽略这些额外字段，只复用其中的 `id/question/response/label`
- `--rerun_tag` 只是追踪字段，方便区分这是不是一轮 drop-only rerun
- 当前推荐把 rerun 输出写到单独文件，再人工检查并决定如何 merge

---

### Step 06: 构建 SFT 训练集

```bash
uv run python scripts/06_build_sft_jsonl.py
```

输出：
- `data/processed/sft_train.jsonl`
- `data/processed/sft_dev.jsonl`

每条样本包含：
- `instruction`
- `input`
- `output`
- `messages`
- `gold_label`

说明：
- `instruction/input/output` 方便新读者直接看懂一条训练样本是什么
- `messages` 方便直接喂给 chat-style SFT 训练代码

---

### Step 07: 训练 Qwen 4B 级结构化输出主模型

这一步是项目的主训练路径，目标是微调 Qwen 4B 级模型，使其稳定输出 `label/reason/evidence` 结构化判决。

```bash
uv run python scripts/07_train_sft_lora.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --train_file data/processed/sft_train.jsonl \
  --eval_file data/processed/sft_dev.jsonl
```

输出：
- adapter: `outputs/sft_lora/`
- 推理模板: `outputs/sft_lora/main_system.txt`
- 推理模板: `outputs/sft_lora/main_user.txt`

如果你使用的是 15000 条版本数据，推荐直接运行：

```bash
uv run python scripts/07_train_sft_lora.py \
  --train_file data/processed/sft_train_15000.jsonl \
  --eval_file data/processed/sft_dev_15000.jsonl \
  --output_dir outputs/sft_lora_15000 \
  --tensorboard_run_name step07_sft_lora_15000 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_seq_length 2048
```

默认使用 LoRA。若环境支持 bitsandbytes，可加：

```bash
uv run python scripts/07_train_sft_lora.py --load_in_4bit
```

#### TensorBoard 监控

step 07 脚本已内置 TensorBoard 上报：
- [scripts/07_train_sft_lora.py:297](scripts/07_train_sft_lora.py#L297)

可另开终端启动：

```bash
uv run tensorboard --logdir outputs/sft_lora_15000/runs --host 0.0.0.0 --port 6006
```

如果 `6006` 已被占用，可换一个空闲端口，例如：

```bash
uv run tensorboard --logdir outputs/sft_lora_15000/runs --host 0.0.0.0 --port 43355
```

#### 当前已知环境冲突：step 04 与 step 07 对 Transformers 的要求不同

这里有一个实际跑通中暴露出来的重要兼容性问题：

1. **step 04 / vLLM teacher 路径**
   - `Qwen/Qwen3.5-27B` 的 vLLM 推理路径更依赖 `vllm` 与 `transformers` 的兼容性
   - 之前为排查 step 04 扩充运行，项目曾临时收紧到 `transformers<5`，以便先把 teacher 扩充跑通
   - teacher 扩充脚本还通过 `enforce_eager=True` 规避了 vLLM 编译/AOT 初始化问题

2. **step 07 / Qwen3.5-4B 训练路径**
   - 如果基座模型的 `config.json` 中：
     - `model_type = "qwen3_5"`
   - `transformers==4.57.6` 无法识别这个架构，训练会报：
     - `KeyError: 'qwen3_5'`
     - `Transformers does not recognize this architecture`

3. **当前推荐修复方案**
   - 直接把项目依赖中的 Transformers 要求恢复到：

```toml
transformers>=5.2.0
```

   - 然后执行：

```bash
uv lock
uv sync
```

   - 同步后再使用正常的 `uv run` 启动 step 07。

4. **当前项目的实际建议**
   - 如果当前重点是 step 07 / 08，优先保证训练环境能加载 `Qwen3.5-4B`
   - 如果之后还要重新跑 step 04 的 vLLM teacher 扩充，再重新检查 `vllm` 与当前 `transformers` 的兼容性
   - 更稳妥的长期方案是把 teacher 推理环境和主模型训练环境拆开，但当前仓库还没有正式做环境隔离

参数理解：
- 默认主模型是 **Qwen 4B 级**，不是 baseline 小模型
- 默认 `per_device_train_batch_size=4 + gradient_accumulation_steps=8`
  适合当前 15000 条版本训练配置
- 在 **A100 80GB** 上，`bf16` 往往是首选；脚本会在可用时优先启用
- `max_seq_length=2048` 适合当前 question/response + 结构化输出长度
- `target_modules` 指的是 transformer 中主要投影层，是 LoRA 的常见注入位置

---

### Step 08: 统一生成与单文件评估

先确认比较口径：

- step 03 baseline 的最终对外评估数据是 [data/processed/test.jsonl](data/processed/test.jsonl)
- baseline 的测试集预测文件是 `outputs/baseline_cls/test_predictions.jsonl`
- step 08 统一把 `test.jsonl` 作为 comparison set，用来产出或评估：
  - step 03 baseline
  - `Qwen/Qwen3.5-4B` base model（SFT 前）
  - step 07 LoRA model（SFT 后）

#### 只评估单个 prediction file

如果你已有生成式模型预测结果文件：

```bash
uv run python scripts/08_eval.py evaluate \
  --prediction_file outputs/predictions.jsonl \
  --reference_file data/processed/test.jsonl \
  --output_file outputs/metrics.json
```

如果只想先评估 baseline 分类器结果：

```bash
uv run python scripts/08_eval.py evaluate \
  --prediction_file outputs/baseline_cls/test_predictions.jsonl \
  --reference_file data/processed/test.jsonl \
  --output_file outputs/baseline_metrics.json
```

#### 只生成 Qwen base / LoRA 在 test.jsonl 上的预测

先生成 base Qwen 预测：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py generate \
  --mode base \
  --test_file data/processed/test.jsonl \
  --output_file outputs/qwen_base_test_predictions.jsonl
```

再生成 step 07 LoRA 预测：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py generate \
  --mode lora \
  --test_file data/processed/test.jsonl \
  --adapter_path outputs/sft_lora \
  --output_file outputs/qwen_lora_test_predictions.jsonl
```

#### 一键顺序跑 base + LoRA 的生成与评估

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py run \
  --mode both \
  --test_file data/processed/test.jsonl \
  --reference_file data/processed/test.jsonl \
  --adapter_path outputs/sft_lora
```

说明：
- step 08 现在只负责 prediction 生成与**单文件评估**，不再承担 compare/report 逻辑。
- `generate` 支持 `--mode base|lora|both`。
- `run` 会先生成，再立刻评估；`both` 模式会顺序执行 base，然后 lora。
- LoRA 推理需要显式传 `--adapter_path`。
- step 08 当前**只使用 Transformers 后端**。这是因为在项目编写期间，Qwen3.5 相关的 vLLM + LoRA 推理支持在本 workflow 中存在兼容性与版本冲突问题，因此这里统一改为 Transformers 推理；step 04 的 teacher 数据生成仍然继续使用 vLLM。
- `both` 模式默认输出：
  - `outputs/qwen_base_test_predictions.jsonl`
  - `outputs/qwen_lora_test_predictions.jsonl`
  - `outputs/qwen_base_metrics.json`
  - `outputs/qwen_lora_metrics.json`
- 如果 `outputs/sft_lora/` 中存在 `main_system.txt` 和 `main_user.txt`，LoRA 推理会优先复用这组训练期导出的 prompt；否则回退到 `prompts/` 下的 canonical prompt。
- 如有多卡，脚本默认选择当前最空闲的卡；也可手动传 `--gpu_id`。
- 长时间 generation 时会持续输出 JSON progress event，包含 `done/total/percent/elapsed/eta`。

输出指标按三组组织：
- `label_metrics`
  - `accuracy`
  - `macro_f1`
  - `unsafe_precision`
  - `unsafe_recall`
  - `unsafe_f1`
  - `overblock_rate`
- `parse_metrics`
  - `raw_json_parse_success_rate`
  - `raw_json_parse_failure_rate`
  - `fallback_usable_rate`
- `rationale_metrics`
  - `evidence_hit_rate`
  - `reason_label_consistency`

说明：
- label 指标适合做主效果比较。
- rationale 指标是启发式质量检查，不等价于人工评估。
- `raw_json_parse_success_rate` 衡量模型原始输出能否被严格解析成 JSON。
- `fallback_usable_rate` 衡量即使原始输出不够规范，是否仍能从现有字段中恢复出可评估结果。
- 为兼容旧报告，输出里仍会保留 `json_valid_rate`，其含义等同于 `fallback_usable_rate`。
- `evidence_hit_rate` 只检查证据是否能在原文定位，不代表理由一定充分。
- 当前 `overblock_rate` 定义为：`gold == safe` 且 `pred == unsafe` 的样本数 / `gold == safe` 的样本数。
- step 08 会校验 prediction file 与 `test.jsonl` 的行数与 `id` 对齐，避免误把不同 split 混在一起比较。

### Step 09: 多模型对比分析

当你已经有 baseline / Qwen base / Qwen LoRA 三份 step-08-compatible prediction file 时，可以使用 step 09 生成统一横向分析报告：

```bash
uv run python scripts/09_compare_models.py \
  --reference_file data/processed/test.jsonl \
  --baseline_file outputs/baseline_cls/test_predictions.jsonl \
  --qwen_base_file outputs/qwen_base_test_predictions.jsonl \
  --qwen_lora_file outputs/qwen_lora_test_predictions.jsonl \
  --output_dir outputs/step09_compare
```

step 09 会输出：
- `comparison_analysis.json`
- `comparison_analysis.md`

step 09 是唯一的 compare / analysis 层，负责：
- confusion matrix
- predicted label distribution
- pairwise delta
- per-metric ranking
- 自动生成的关键结论（基于真实结果，而不是写死假设）


## 4. 常见错误

### 1) 无法下载 Hugging Face 模型或数据集

先确认环境变量：

```bash
echo $HF_ENDPOINT
echo $HF_HOME
```

建议设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.hf-cache
```

如果你使用的是 `uv run ...`，也建议先执行：

```bash
set -a
source .env
set +a
```

### 2) 4bit 量化报错

请确认安装了 `bitsandbytes`，且 CUDA 环境可用。没有 GPU 时先不要加 `--load_in_4bit`。

### 3) 数据格式错误 / JSON 解析失败

先依次重跑：

```bash
uv run python scripts/04_generate_rationale_pseudo.py --max_samples 20
uv run python scripts/05_filter_pseudo_labels.py
uv run python scripts/06_build_sft_jsonl.py
```

用小样本检查 teacher 输出格式是否稳定。

### 4) OOM

baseline 路径优先降低：
- `--per_device_train_batch_size`
- `--max_length`

Qwen 主路径优先降低：
- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--max_seq_length`
- 或开启 `--load_in_4bit`

---

## 5. 数据集情况

以下统计基于 **原始** `allenai/wildguardmix` 数据集的官方 `train/dev/test` split，统计时间为 **2026-03-27**。它主要用于帮助读者快速理解原始数据分布，不代表后续经过过滤、重采样或伪标注处理后的项目中间数据规模。

- `train`：共 34140 条，其中 `safe=26609`、`unsafe=7531`，占比约为 `77.9% / 22.1%`
- `dev`：共 3794 条，其中 `safe=2957`、`unsafe=837`，占比约为 `77.9% / 22.1%`
- `test`：共 1709 条，其中 `safe=1425`、`unsafe=284`，占比约为 `83.4% / 16.6%`
