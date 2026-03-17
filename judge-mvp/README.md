# Judge MVP: WildGuardMix 内容安全裁判模型

这个项目从零搭建一个可执行的 LLM 后训练 MVP，目标是基于 `allenai/wildguardmix` 训练一个输出 `label + reason + evidence + confidence` 的内容安全裁判模型。

## 目录结构

```text
judge-mvp/
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
cd judge-mvp
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
     `label/reason/evidence/confidence` 的结构化安全裁判模型。
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
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --input_path data/processed/train.jsonl \
  --max_samples 200
```

输出：
- `data/interim/pseudo_raw.jsonl`

teacher 目标输出 JSON：

```json
{
  "label": "safe",
  "reason": "...",
  "evidence": ["..."],
  "confidence": 0.92
}
```

说明：
- 这一步是主路径的开始，不再只是分类，而是开始构造结构化监督信号。
- 脚本会同时保留原始 teacher 文本和解析后的 JSON，便于后续调试解析失败问题。
- 建议先用小样本验证流程。

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
- 剔除常见拒答模板

输出：
- `data/processed/pseudo_filtered.jsonl`

为什么要做这一步：
- 结构化 SFT 对噪声更敏感
- 错误 evidence / 空 reason / 拒答式输出会显著拉低训练质量
- 与其追求样本更多，不如先保证监督信号更干净

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

这一步是项目的主训练路径，目标是微调 Qwen 4B 级模型，使其稳定输出 `label/reason/evidence/confidence` 结构化判决。

```bash
uv run python scripts/07_train_sft_lora.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --train_file data/processed/sft_train.jsonl \
  --eval_file data/processed/sft_dev.jsonl
```

输出：
- adapter: `outputs/sft_lora/`
- 推理模板: `outputs/sft_lora/prompt_template.txt`

默认使用 LoRA。若环境支持 bitsandbytes，可加：

```bash
uv run python scripts/07_train_sft_lora.py --load_in_4bit
```

参数理解：
- 默认主模型是 **Qwen 4B 级**，不是 baseline 小模型
- 默认 `per_device_train_batch_size=1 + gradient_accumulation_steps=8`
  是为了让脚本在学习阶段更稳妥，也更接近大模型常见训练方式
- 在 **A100 80GB** 上，`bf16` 往往是首选；脚本会在可用时优先启用
- `max_seq_length=1024` 是一个便于起步的默认值，通常足以覆盖
  question/response 加结构化输出
- `target_modules` 指的是 transformer 中主要投影层，是 LoRA 的常见注入位置

---

### Step 08: 统一评估

如果你已有生成式模型预测结果文件：

```bash
uv run python scripts/08_eval.py \
  --prediction_file outputs/predictions.jsonl \
  --output_file outputs/metrics.json
```

如果只想先评估 baseline 分类器结果：

```bash
uv run python scripts/08_eval.py \
  --prediction_file outputs/baseline_cls/test_predictions.jsonl \
  --output_file outputs/metrics.json
```

输出指标包含：
- `macro_f1`
- `unsafe_recall`
- `overblock_rate`
- `refusal_rate`
- `json_valid_rate`
- `evidence_hit_rate`
- `reason_label_consistency`

说明：
- label 指标适合做主效果比较
- rationale 指标是启发式质量检查，不等价于人工评估
- `evidence_hit_rate` 只检查证据是否能在原文定位，不代表理由一定充分

---

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

## 5. 验证清单

1. 环境验证：
   - `uv run python scripts/01_inspect_dataset.py` 能打印字段与样本。
2. 数据验证：
   - `data/processed/*.jsonl` 存在且 label 只包含 `safe/unsafe`。
3. baseline 验证：
   - `uv run python scripts/03_train_baseline_cls.py --num_train_epochs 1` 能产出：
     - `outputs/baseline_cls/metrics.json`
     - `outputs/baseline_cls/test_predictions.jsonl`
4. 主路径验证：
   - `README` 与 `scripts/07_train_sft_lora.py` 都明确主模型是 Qwen 4B 级，baseline 不是最终模型。
5. 输出验证：
   - `outputs/metrics.json` 包含：
     - `macro_f1`
     - `unsafe_recall`
     - `overblock_rate`
     - `refusal_rate`
     - `json_valid_rate`
     - `evidence_hit_rate`
6. 端到端抽检：
   - 按 `01 -> 03` 至少跑通一次
   - 随机查看预测结果，确认脚本输入输出和设计目的都能从 README 与源码 docstring 中读懂
