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

推荐 Python 3.10+，并在项目内使用 `uv`。

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

本项目默认提供以下镜像友好配置：
- `uv.toml`：默认使用清华 PyPI 镜像
- `pyproject.toml`：配置 `tool.uv.index` 与 `tool.uv.pip.index-url`
- `.env.example`：提供 `HF_ENDPOINT`、`HF_HOME`、`PIP_INDEX_URL`、`UV_INDEX_URL`

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

## 1. 一键跑通顺序

按下面顺序运行：

```bash
python scripts/01_inspect_dataset.py
python scripts/02_build_binary_dataset.py
python scripts/03_train_baseline_cls.py
python scripts/04_generate_rationale_pseudo.py
python scripts/05_filter_pseudo_labels.py
python scripts/06_build_sft_jsonl.py
python scripts/07_train_sft_lora.py
python scripts/08_eval.py
```

---

## 2. 各步骤说明

### Step 01: 查看原始数据集结构

```bash
python scripts/01_inspect_dataset.py
```

输出内容：
- dataset splits
- 每个 split 的 features
- 样本预览

用途：先确认 `wildguardmix` 实际字段，避免后续 schema 假设错误。

---

### Step 02: 构建二分类标准数据集

```bash
python scripts/02_build_binary_dataset.py
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

---

### Step 03: 训练只看标签的 baseline 分类器

```bash
python scripts/03_train_baseline_cls.py \
  --model_name_or_path distilbert-base-uncased \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8
```

输入：`question + response`

输出：
- checkpoint: `outputs/baseline_cls/`
- 指标文件: `outputs/baseline_cls/metrics.json`

关键指标：
- `accuracy`
- `macro_f1`
- `unsafe_precision`
- `unsafe_recall`

---

### Step 04: 生成理由+证据伪标注

```bash
python scripts/04_generate_rationale_pseudo.py \
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

建议先用小样本验证流程。

---

### Step 05: 过滤伪标注

```bash
python scripts/05_filter_pseudo_labels.py
```

过滤规则：
- teacher label 必须与金标签一致
- `evidence` 至少 1 条
- 每条 evidence 必须能在原始 `question` 或 `response` 中匹配
- `reason` 长度合理
- 剔除常见拒答模板

输出：
- `data/processed/pseudo_filtered.jsonl`

---

### Step 06: 构建 SFT 训练集

```bash
python scripts/06_build_sft_jsonl.py
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

---

### Step 07: 训练结构化输出 SFT 模型

```bash
python scripts/07_train_sft_lora.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_file data/processed/sft_train.jsonl \
  --eval_file data/processed/sft_dev.jsonl
```

输出：
- adapter: `outputs/sft_lora/`
- 推理模板: `outputs/sft_lora/prompt_template.txt`

默认使用 LoRA。若环境支持 bitsandbytes，可加：

```bash
python scripts/07_train_sft_lora.py --load_in_4bit
```

---

### Step 08: 统一评估

如果你已有生成式模型预测结果文件：

```bash
python scripts/08_eval.py \
  --prediction_file outputs/predictions.jsonl \
  --output_file outputs/metrics.json
```

如果只想先评估 baseline 分类器结果：

```bash
python scripts/08_eval.py \
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

---

## 3. 常见错误

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

### 2) 4bit 量化报错

请确认安装了 `bitsandbytes`，且 CUDA 环境可用。没有 GPU 时先不要加 `--load_in_4bit`。

### 3) 数据格式错误 / JSON 解析失败

先依次重跑：

```bash
python scripts/04_generate_rationale_pseudo.py --max_samples 20
python scripts/05_filter_pseudo_labels.py
python scripts/06_build_sft_jsonl.py
```

用小样本检查 teacher 输出格式是否稳定。

### 4) OOM

优先降低：
- `--per_device_train_batch_size`
- `--max_length`
- `--max_samples`

或改用更小模型先打通流程。

---

## 4. 验证清单

1. 环境验证：
   - `python scripts/01_inspect_dataset.py` 能打印字段与样本。
2. 数据验证：
   - `data/processed/*.jsonl` 存在且 label 只包含 `safe/unsafe`。
3. 训练验证：
   - baseline 与 SFT 能产出 checkpoint/log，不报 schema 错误。
4. 输出验证：
   - `outputs/metrics.json` 包含：
     - `macro_f1`
     - `unsafe_recall`
     - `overblock_rate`
     - `refusal_rate`
     - `json_valid_rate`
     - `evidence_hit_rate`
5. 端到端抽检：
   - 随机查看预测结果，确认能稳定输出结构化判决，并给出可在原文中定位的证据片段。
