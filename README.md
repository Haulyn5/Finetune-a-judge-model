# Finetune-a-judge-model

<p align="center">
  <strong>基于 WildGuardMix 的内容安全裁判模型训练 MVP</strong><br />
  从二分类 baseline 到结构化 <code>label + reason + evidence</code> 输出的完整后训练流程
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img alt="status" src="https://img.shields.io/badge/Status-MVP-orange" />
  <img alt="workflow" src="https://img.shields.io/badge/Workflow-01--09-success" />
  <img alt="model" src="https://img.shields.io/badge/Base-Qwen3.5--4B-black" />
</p>

本项目从零搭建了一个可执行的 LLM 后训练流程，目标是基于 `allenai/wildguardmix` 训练一个内容安全裁判模型。它不仅判断 `safe / unsafe`，还尽量输出可解释的结构化结果：

```json
{
  "label": "unsafe",
  "reason": "The response provides harmful guidance.",
  "evidence": ["..."]
}
```

仓库包括一条完整、可复现、可分析的实验路径：

- `01 -> 03`：快速建立数据理解和轻量 baseline
- `04 -> 08`：构建 teacher 伪标注、过滤、SFT、统一评估
- `09`：做多模型横向比较和分析报告

## Overview

### 这个仓库适合什么场景

- 本项目源自作者对 Vibe Coding 实践与 LLM 微调的学习探索，适合面向大语言模型安全方向的学习者快速搭建后训练全流程。

### 核心特性

- 基于 `WildGuardMix` 构建统一数据接口
- 对比了 Qwen 3.5-4B 的 base model 与 LoRA 模型效果 以及基于 BERT 的轻量 baseline
- 使用 Qwen 3.5-27B teacher 生成结构化伪标注，再做过滤
- 支持 LoRA / 可选 QLoRA 风格训练
- 提供统一 step-08 评估入口和 step-09 对比分析
- 运行中输出 JSON progress event，适合长任务监控

### 当前项目状态

这是一个面向学习和实验验证的 MVP，目前已经覆盖：

- 数据检查
- 二分类标准化
- baseline 分类训练
- teacher 结构化伪标注
- 伪标注过滤
- SFT 数据构建
- Qwen LoRA 训练
- 单模型评估
- 多模型比较分析

## Quick Start

### 1. 环境准备

推荐 Python `3.10+`，默认使用 `uv` 管理环境。

```bash
cp .env.example .env
set -a
source .env
set +a

uv venv
source .venv/bin/activate
uv sync
```

如果还没有安装 `uv`，请先按官方方式安装后再执行上述命令。

### 2. 一键跑通主流程

```bash
uv run python scripts/01_inspect_dataset.py
uv run python scripts/02_build_binary_dataset.py
uv run python scripts/03_train_baseline_cls.py
uv run python scripts/04_generate_rationale_pseudo.py
uv run python scripts/05_filter_pseudo_labels.py
uv run python scripts/06_build_sft_jsonl.py
uv run python scripts/07_train_sft_lora.py
uv run python scripts/08_eval.py run --mode both --adapter_path outputs/sft_lora
uv run python scripts/09_compare_models.py \
  --reference_file data/processed/test.jsonl \
  --baseline_file outputs/baseline_cls/test_predictions.jsonl \
  --qwen_base_file outputs/qwen_base_test_predictions.jsonl \
  --qwen_lora_file outputs/qwen_lora_test_predictions.jsonl \
  --output_dir outputs/step09_compare
```

### 3. 第一次阅读建议

如果你是第一次接触这个项目，推荐按下面顺序理解：

1. 先看 `01 -> 03`，理解数据 schema、标签口径、baseline 指标。
2. 再看 `04 -> 06`，理解结构化伪标注是如何生成和过滤的。
3. 最后看 `07 -> 09`，理解主模型训练、评估与比较。

## Pipeline

### 两条实验线

| 路线 | 步骤 | 目标 | 备注 |
| --- | --- | --- | --- |
| Baseline 线 | `01 -> 03` | 快速建立 `safe/unsafe` 参照 | 用于对照，不是最终模型 |
| 主训练线 | `04 -> 08` | 训练结构化安全裁判模型 | 目标输出 `label/reason/evidence` |
| 分析线 | `09` | 统一比较多个模型结果 | 输出 markdown 和 json 报告 |

### 端到端流程图

```text
WildGuardMix
   |
   v
01 inspect dataset
   |
   v
02 build binary dataset --------------------> 03 baseline classifier
   |                                              |
   |                                              v
   +--> 04 teacher pseudo labels -> 05 filter -> baseline predictions
                                |
                                v
                         06 build SFT jsonl
                                |
                                v
                         07 Qwen LoRA training
                                |
                                v
                         08 generation + eval
                                |
                                v
                         09 comparison report
```

## Repository Layout

```text
Finetune-a-judge-model/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notes/
├── outputs/
├── prompts/
├── scripts/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### 关键目录说明

| 路径 | 作用 |
| --- | --- |
| `scripts/` | 所有 step 脚本和共享工具函数 |
| `data/raw/` | 原始数据占位 |
| `data/interim/` | teacher 原始输出和中间文件 |
| `data/processed/` | 标准化后的训练/验证/测试与 SFT 数据 |
| `outputs/` | checkpoint、预测结果、指标、报告 |
| `prompts/` | teacher/main 模型提示词模板 |
| `notes/` | 实验过程中的分析记录 |

## Environment Notes

### 中国大陆网络环境建议

项目已经对镜像和缓存做了默认配置：

- `pyproject.toml`：配置了 `tool.uv.index-url` 与 `extra-index-url`
- `.env.example`：提供 `HF_ENDPOINT`、`HF_HOME`、`PIP_INDEX_URL`、`UV_INDEX_URL`
- `.hf-cache/`：默认作为 Hugging Face 缓存目录
- `uv.lock`：锁定当前依赖解析结果，便于复现

如需手动导出环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.hf-cache
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

如果仍需使用 `pip`：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

脚本不会硬编码代理或镜像地址，只读取当前 shell 环境变量。

## Step-by-Step

<details>
<summary><strong>Step 01. 查看原始数据集结构</strong></summary>

运行：

```bash
uv run python scripts/01_inspect_dataset.py
```

输出内容：

- dataset splits
- 每个 split 的 features
- 样本预览

用途：

- 先确认 `wildguardmix` 的真实字段
- 避免后续 schema 假设错误

</details>

<details>
<summary><strong>Step 02. 构建二分类标准数据集</strong></summary>

运行：

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
- 使用 `response_harm_label` 映射为 `safe / unsafe`
- 自动过滤缺失值与异常标签
- 从训练集切出开发集

这是后续所有脚本共享的数据接口层，建议在继续之前人工抽查输出样本。

</details>

<details>
<summary><strong>Step 03. 训练轻量 baseline 分类器</strong></summary>

这一步只用于建立对照指标，不是最终结构化裁判模型。

```bash
uv run python scripts/03_train_baseline_cls.py \
  --model_name_or_path distilbert-base-uncased \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8
```

输入：

- `data/processed/{train,dev,test}.jsonl`

输出：

- checkpoint：`outputs/baseline_cls/`
- 指标：`outputs/baseline_cls/metrics.json`
- 测试集预测：`outputs/baseline_cls/test_predictions.jsonl`

关键指标：

- `accuracy`
- `macro_f1`
- `unsafe_precision`
- `unsafe_recall`
- `unsafe_f1`

</details>

<details>
<summary><strong>Step 04. 生成理由 + 证据伪标注</strong></summary>

```bash
uv run python scripts/04_generate_rationale_pseudo.py \
  --teacher_model Qwen/Qwen3.5-27B \
  --input_path data/processed/train.jsonl \
  --max_samples 3000 \
  --sampling_strategy balanced
```

输出：

- `data/interim/pseudo_raw.jsonl`

teacher 目标输出：

```json
{
  "label": "safe",
  "reason": "...",
  "evidence": ["..."]
}
```

说明：

- 这是主路径真正开始“结构化监督”的地方
- 默认会优先做 `safe / unsafe` 平衡采样
- 保留原始 teacher 文本和解析结果，方便后续调试
- 当前 teacher 路径使用 `vLLM`

</details>

<details>
<summary><strong>Step 05. 过滤伪标注</strong></summary>

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

设计动机：

- 结构化 SFT 对噪声比普通分类更敏感
- 与其盲目扩样本，不如优先提高监督质量

#### 定向重跑 dropped 子集

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

</details>

<details>
<summary><strong>Step 06. 构建 SFT 训练集</strong></summary>

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

其中：

- `instruction/input/output` 方便直接阅读样本
- `messages` 方便直接喂给 chat-style SFT 训练代码

</details>

<details>
<summary><strong>Step 07. 训练 Qwen 4B 级结构化主模型</strong></summary>

```bash
uv run python scripts/07_train_sft_lora.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --train_file data/processed/sft_train.jsonl \
  --eval_file data/processed/sft_dev.jsonl
```

输出：

- adapter：`outputs/sft_lora/`
- 推理模板：`outputs/sft_lora/main_system.txt`
- 推理模板：`outputs/sft_lora/main_user.txt`

15000 条版本的推荐命令：

```bash
uv run python scripts/07_train_sft_lora.py \
  --train_file data/processed/sft_train_15000.jsonl \
  --eval_file data/processed/sft_dev_15000.jsonl \
  --output_dir outputs/sft_lora_15000 \
  --tensorboard_run_name step07_sft_lora_15000 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_seq_length 2048
```

可选 4bit：

```bash
uv run python scripts/07_train_sft_lora.py --load_in_4bit
```

TensorBoard：

```bash
uv run tensorboard --logdir outputs/sft_lora_15000/runs --host 0.0.0.0 --port 6006
```

默认设计假设：

- 主路径对齐 `Qwen 4B` 级模型
- 训练环境建议至少接近 `A100 80GB`
- 优先使用 `bf16`，可选 LoRA / 4bit

</details>

<details>
<summary><strong>Step 08. 统一生成与单文件评估</strong></summary>

step 08 是统一入口，负责：

- 生成 base model 预测
- 生成 LoRA model 预测
- 评估单个 prediction file
- 顺序执行 generate + evaluate

只评估一个 prediction file：

```bash
uv run python scripts/08_eval.py evaluate \
  --prediction_file outputs/predictions.jsonl \
  --reference_file data/processed/test.jsonl \
  --output_file outputs/metrics.json
```

只生成 base Qwen 预测：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py generate \
  --mode base \
  --test_file data/processed/test.jsonl \
  --output_file outputs/qwen_base_test_predictions.jsonl
```

只生成 LoRA 预测：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py generate \
  --mode lora \
  --test_file data/processed/test.jsonl \
  --adapter_path outputs/sft_lora \
  --output_file outputs/qwen_lora_test_predictions.jsonl
```

一键跑 base + LoRA 的生成与评估：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/08_eval.py run \
  --mode both \
  --test_file data/processed/test.jsonl \
  --reference_file data/processed/test.jsonl \
  --adapter_path outputs/sft_lora
```

step 08 当前统一使用 `Transformers` 后端；teacher 数据生成仍使用 `vLLM`。

</details>

<details>
<summary><strong>Step 09. 多模型对比分析</strong></summary>

```bash
uv run python scripts/09_compare_models.py \
  --reference_file data/processed/test.jsonl \
  --baseline_file outputs/baseline_cls/test_predictions.jsonl \
  --qwen_base_file outputs/qwen_base_test_predictions.jsonl \
  --qwen_lora_file outputs/qwen_lora_test_predictions.jsonl \
  --output_dir outputs/step09_compare
```

输出：

- `comparison_analysis.json`
- `comparison_analysis.md`

step 09 负责：

- confusion matrix
- predicted label distribution
- pairwise delta
- per-metric ranking
- 自动生成关键结论

</details>

## Evaluation

### step 08 输出指标

| 指标组 | 代表字段 | 说明 |
| --- | --- | --- |
| `label_metrics` | `accuracy`, `macro_f1`, `unsafe_f1`, `overblock_rate` | 用于主效果比较 |
| `parse_metrics` | `raw_json_parse_success_rate`, `fallback_usable_rate` | 衡量结构化输出可解析性 |
| `rationale_metrics` | `evidence_hit_rate`, `reason_label_consistency` | 启发式检查解释质量 |

### 评估口径说明

- step 03 baseline 的统一评估集是 `data/processed/test.jsonl`
- step 08 会校验 prediction file 与 `test.jsonl` 的行数和 `id` 对齐
- `json_valid_rate` 为兼容旧报告仍会保留，含义等同于 `fallback_usable_rate`
- `overblock_rate` 当前定义为：
  `gold == safe` 且 `pred == unsafe` 的样本数 / `gold == safe` 的样本数

## Known Limitations

### 1. step 04 与 step 07 的依赖兼容性存在张力

这是当前仓库最重要的工程限制之一：

- step 04 的 teacher 推理依赖 `vllm` 与 `transformers` 的兼容性
- step 07 的 Qwen3.5-4B 训练路径需要较新的 `transformers`
- 项目当前建议优先保证 step 07 / 08 可运行

当前 `pyproject.toml` 中推荐保持：

```toml
transformers>=5.2.0
```

然后执行：

```bash
uv lock
uv sync
```

更稳妥的长期方案是将 teacher 推理环境与主模型训练环境拆分，但当前仓库还没有正式环境隔离。

### 2. 硬件假设偏高

- baseline 路径较轻量
- Qwen 主路径默认按较强 GPU 环境设计
- `--load_in_4bit` 只能缓解一部分显存压力，不能完全替代合理硬件

### 3. rationale 指标仍是启发式

`evidence_hit_rate` 与 `reason_label_consistency` 只能辅助观察，不等价于人工质量评测。

## FAQ

### 1. 无法下载 Hugging Face 模型或数据集

先确认：

```bash
echo $HF_ENDPOINT
echo $HF_HOME
```

建议设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.hf-cache
```

如果你使用的是 `uv run ...`，建议先执行：

```bash
set -a
source .env
set +a
```

### 2. 4bit 量化报错

请确认：

- 已安装 `bitsandbytes`
- CUDA 环境可用
- 没有 GPU 时不要加 `--load_in_4bit`

### 3. 数据格式错误或 JSON 解析失败

建议小样本回归：

```bash
uv run python scripts/04_generate_rationale_pseudo.py --max_samples 20
uv run python scripts/05_filter_pseudo_labels.py
uv run python scripts/06_build_sft_jsonl.py
```

### 4. OOM

baseline 路径优先降低：

- `--per_device_train_batch_size`
- `--max_length`

Qwen 主路径优先降低：

- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--max_seq_length`
- 或开启 `--load_in_4bit`

## Dataset Snapshot

以下统计基于原始 `allenai/wildguardmix` 官方 `train/dev/test` split，统计时间为 `2026-03-27`。这些数字用于帮助快速理解原始数据分布，不代表经过过滤、重采样或伪标注处理后的项目中间数据规模。

| Split | Total | Safe | Unsafe | Safe Ratio | Unsafe Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 34140 | 26609 | 7531 | 77.9% | 22.1% |
| `dev` | 3794 | 2957 | 837 | 77.9% | 22.1% |
| `test` | 1709 | 1425 | 284 | 83.4% | 16.6% |

## Outputs You Should Expect

跑完整条流程后，常见产物包括：

- `data/processed/train.jsonl`
- `data/processed/dev.jsonl`
- `data/processed/test.jsonl`
- `data/interim/pseudo_raw.jsonl`
- `data/processed/pseudo_filtered.jsonl`
- `data/processed/sft_train.jsonl`
- `outputs/baseline_cls/`
- `outputs/sft_lora/`
- `outputs/qwen_base_metrics.json`
- `outputs/qwen_lora_metrics.json`
- `outputs/step09_compare/comparison_analysis.md`

## Notes

`notes/` 目录保存了作者的过程性实验记录，希望能为读者提供一些 Insights。

## Roadmap

当前更值得补的方向：

- 增加 `CONTRIBUTING.md`
- 拆分 teacher 与 training 环境
- 补充更稳定的 benchmark 和结果表
