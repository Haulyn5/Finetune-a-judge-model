# Step 09 模型对比分析总结

## 本次运行概览

- 参考集：`data/processed/test.jsonl`
- 样本数：1709
- 标签分布：`safe=1425`，`unsafe=284`
- 对比对象：`baseline`、`qwen_base`、`qwen_lora`

## 新增实验记录：LoRA checkpoint-654

### step 08 生成参数

- 命令：`CUDA_VISIBLE_DEVICES=2 uv run python scripts/08_eval.py generate --mode lora --model_name_or_path /root/project/PretrainedModels/Qwen/Qwen3.5-4B --adapter_path outputs/sft_lora_15000/checkpoint-654 --test_file data/processed/test.jsonl --output_file outputs/qwen_lora_checkpoint654_test_predictions.jsonl --overwrite --gpu_id 0 --transformers_batch_size 32 --max_new_tokens 4096`
- base model：`/root/project/PretrainedModels/Qwen/Qwen3.5-4B`
- adapter checkpoint：`outputs/sft_lora_15000/checkpoint-654`
- mode：`lora`
- backend：`transformers`
- `CUDA_VISIBLE_DEVICES=2`，脚本内 `--gpu_id 0`
- `transformers_batch_size=32`
- `max_new_tokens=4096`
- 输出文件：`outputs/qwen_lora_checkpoint654_test_predictions.jsonl`

### checkpoint-654 分类指标

| Run | accuracy | macro_f1 | unsafe_precision | unsafe_recall | unsafe_f1 | overblock_rate |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_lora_checkpoint654 | 0.9193 | 0.8587 | 0.7386 | 0.7958 | 0.7661 | 0.0561 |

### checkpoint-654 混淆矩阵

- `safe -> safe`: 1345
- `safe -> unsafe`: 80
- `unsafe -> safe`: 58
- `unsafe -> unsafe`: 226

### checkpoint-654 与之前 LoRA 结果对比

| Run | accuracy | macro_f1 | unsafe_precision | unsafe_recall | unsafe_f1 | overblock_rate | raw_json_parse_success_rate | evidence_hit_rate | reason_label_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_lora（之前） | 0.9140 | 0.8414 | 0.7546 | 0.7148 | 0.7342 | 0.0463 | 0.9988 | 0.9105 | 0.9274 |
| qwen_lora_checkpoint654 | 0.9193 | 0.8587 | 0.7386 | 0.7958 | 0.7661 | 0.0561 | 0.9982 | 0.9473 | 0.8484 |
| delta（654 - 之前） | +0.0053 | +0.0172 | -0.0161 | +0.0810 | +0.0319 | +0.0098 | -0.0006 | +0.0369 | -0.0790 |

解读：

- `checkpoint-654` 的主收益来自 **unsafe recall** 提升：`0.7148 -> 0.7958`，因此 `unsafe_f1` 和 `macro_f1` 都进一步上涨。
- 代价是更激进地预测 `unsafe`：`safe -> unsafe` 从 66 增加到 80，`overblock_rate` 从 `0.0463` 升到 `0.0561`。
- 结构化输出稳定性依然很高，但 `reason_label_consistency` 较之前 LoRA 明显下降：`0.9274 -> 0.8484`。
- 如果当前目标更偏向 **尽量抓住 unsafe 样本**，那么 `checkpoint-654` 优于之前 LoRA；如果更重视 **误杀控制和 explanation 一致性**，之前 LoRA 仍有优势。

## 新增实验记录：两组新 LoRA 训练

### 训练参数

#### 1) lr=5e-5 completion_only

- 训练命令：`cd "/root/project/learnTrainLLM" && CUDA_VISIBLE_DEVICES=2 uv run python scripts/07_train_sft_lora.py --train_file data/processed/sft_train_15000.jsonl --eval_file data/processed/sft_dev_15000.jsonl --output_dir outputs/sft_lora_15000_lr_5e-5_completion_only --tensorboard_run_name step07_sft_lora_15000 --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 5e-5 --max_seq_length 2048`
- 评估命令：`uv run python scripts/08_eval.py generate --mode lora --model_name_or_path /root/project/PretrainedModels/Qwen/Qwen3.5-4B --adapter_path outputs/sft_lora_15000_lr_5e-5_completion_only --test_file data/processed/test.jsonl --output_file outputs/qwen_lora_lr_5e-5_completion_only_test_predictions.jsonl --overwrite --gpu_id 1 --transformers_batch_size 32 --max_new_tokens 4096`
- 训练产物目录：`outputs/sft_lora_15000_lr_5e-5_completion_only`
- step 08 输出：`outputs/qwen_lora_lr_5e-5_completion_only_test_predictions.jsonl`

#### 2) lr=3e-5 r32

- 训练命令：`cd "/root/project/learnTrainLLM" && CUDA_VISIBLE_DEVICES=2 uv run python scripts/07_train_sft_lora.py --train_file data/processed/sft_train_15000.jsonl --eval_file data/processed/sft_dev_15000.jsonl --output_dir outputs/sft_lora_15000_lr_3e-5_r32 --tensorboard_run_name step07_sft_lora_15000 --num_train_epochs 4 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --max_seq_length 2048`
- 评估命令：`uv run python scripts/08_eval.py generate --mode lora --model_name_or_path /root/project/PretrainedModels/Qwen/Qwen3.5-4B --adapter_path outputs/sft_lora_15000_lr_3e-5_r32 --test_file data/processed/test.jsonl --output_file outputs/qwen_lora_lr_3e-5_r32_test_predictions.jsonl --overwrite --gpu_id 0 --transformers_batch_size 32 --max_new_tokens 4096`
- 训练产物目录：`outputs/sft_lora_15000_lr_3e-5_r32`
- step 08 输出：`outputs/qwen_lora_lr_3e-5_r32_test_predictions.jsonl`

### 两组新 LoRA 的 test split 指标

| Run | accuracy | macro_f1 | unsafe_precision | unsafe_recall | unsafe_f1 | overblock_rate | raw_json_parse_success_rate | evidence_hit_rate | reason_label_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr=5e-5 completion_only | 0.9198 | 0.8583 | 0.7458 | 0.7852 | 0.7650 | 0.0533 | 1.0000 | 0.9350 | 0.8631 |
| lr=3e-5 r32 | 0.9163 | 0.8517 | 0.7374 | 0.7711 | 0.7539 | 0.0547 | 0.9994 | 0.9468 | 0.8543 |

### 两组新 LoRA 的混淆矩阵

#### lr=5e-5 completion_only

- `safe -> safe`: 1349
- `safe -> unsafe`: 76
- `unsafe -> safe`: 61
- `unsafe -> unsafe`: 223

#### lr=3e-5 r32

- `safe -> safe`: 1347
- `safe -> unsafe`: 78
- `unsafe -> safe`: 65
- `unsafe -> unsafe`: 219

### 两组新 LoRA 结论

- 在这两组新实验里，**`lr=5e-5 completion_only` 综合表现更好**。
- 它在 `accuracy`、`macro_f1`、`unsafe_precision`、`unsafe_recall`、`unsafe_f1` 上都高于 `lr=3e-5 r32`，同时 `overblock_rate` 更低。
- `lr=3e-5 r32` 的唯一明显优势是 `evidence_hit_rate` 略高，但整体分类表现不如 `lr=5e-5 completion_only`。
- `lr=5e-5 completion_only` 的 `raw_json_parse_success_rate = 1.0000`，说明这组输出在结构化 JSON 稳定性上也非常强。

## LoRA 各实验总表

| Run | adapter / output | accuracy | macro_f1 | unsafe_precision | unsafe_recall | unsafe_f1 | overblock_rate | raw_json_parse_success_rate | evidence_hit_rate | reason_label_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_lora（原始） | `outputs/qwen_lora_test_predictions.jsonl` | 0.9140 | 0.8414 | **0.7546** | 0.7148 | 0.7342 | **0.0463** | 0.9988 | 0.9105 | **0.9274** |
| qwen_lora_checkpoint654 | `outputs/sft_lora_15000/checkpoint-654` | 0.9193 | **0.8587** | 0.7386 | **0.7958** | **0.7661** | 0.0561 | 0.9982 | **0.9473** | 0.8484 |
| lr=5e-5 completion_only | `outputs/sft_lora_15000_lr_5e-5_completion_only` | **0.9198** | 0.8583 | 0.7458 | 0.7852 | 0.7650 | 0.0533 | **1.0000** | 0.9350 | 0.8631 |
| lr=3e-5 r32 | `outputs/sft_lora_15000_lr_3e-5_r32` | 0.9163 | 0.8517 | 0.7374 | 0.7711 | 0.7539 | 0.0547 | 0.9994 | 0.9468 | 0.8543 |

### 总表解读

- 如果按 **accuracy** 看，当前最好的是 `lr=5e-5 completion_only`（`0.9198`）。
- 如果按 **macro_f1 / unsafe_f1 / unsafe_recall** 看，当前最好的是 `checkpoint-654`。
- 如果按 **unsafe_precision / overblock_rate / reason-label consistency** 看，原始 `qwen_lora` 仍然最稳。
- 如果按 **raw_json_parse_success_rate** 看，`lr=5e-5 completion_only` 达到了 `1.0000`，结构化输出最稳定。
- 因此当前几个 LoRA 版本形成了比较明确的取舍：
  - `checkpoint-654`：更偏 **抓 unsafe**，召回最强。
  - `lr=5e-5 completion_only`：更偏 **综合均衡**，accuracy 最好，JSON 最稳。
  - 原始 `qwen_lora`：更偏 **保守稳健**，误杀更低、解释一致性最好。

## 核心结论

- 本次 step 09 中，**`qwen_lora` 是综合表现最好的模型**。
- `qwen_base` 相比 baseline 在 `unsafe_recall` 和 `unsafe_f1` 上有提升，但 `overblock_rate` 明显变高。
- `qwen_lora` 不仅在分类指标上领先，而且在结构化输出稳定性上也远优于 `qwen_base`。

## 主要指标对比

| Model | accuracy | macro_f1 | unsafe_precision | unsafe_recall | unsafe_f1 | overblock_rate |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8830 | 0.7610 | 0.7059 | 0.5070 | 0.5902 | 0.0421 |
| qwen_base | 0.8771 | 0.7783 | 0.6303 | 0.6303 | 0.6303 | 0.0737 |
| qwen_lora | 0.9140 | 0.8414 | 0.7546 | 0.7148 | 0.7342 | 0.0463 |

解读：

- `baseline` 的优点是误杀较少，但漏掉了较多 `unsafe` 样本。
- `qwen_base` 更激进，更愿意预测为 `unsafe`，因此召回上升，但误杀也显著增加。
- `qwen_lora` 在召回、精度、F1 和总体准确率之间取得了最好的平衡。

## 混淆矩阵解读

### baseline

- `safe -> safe`: 1365
- `safe -> unsafe`: 60
- `unsafe -> safe`: 140
- `unsafe -> unsafe`: 144

说明：误杀少，但漏判 `unsafe` 很多。

### qwen_base

- `safe -> safe`: 1320
- `safe -> unsafe`: 105
- `unsafe -> safe`: 105
- `unsafe -> unsafe`: 179

说明：比 baseline 抓到了更多 `unsafe`，但对 `safe` 的误杀明显增多。

### qwen_lora

- `safe -> safe`: 1359
- `safe -> unsafe`: 66
- `unsafe -> safe`: 81
- `unsafe -> unsafe`: 203

说明：既减少了 `unsafe` 漏判，又把误杀控制在接近 baseline 的水平，整体最优。

## Pairwise 对比

### qwen_base vs baseline

- `macro_f1`: `+0.0173`
- `unsafe_f1`: `+0.0401`
- `overblock_rate`: `+0.0316`

说明：`qwen_base` 的提升主要来自更高的 `unsafe` 召回，但代价是 overblock 明显增加。

### qwen_lora vs baseline

- `macro_f1`: `+0.0805`
- `unsafe_f1`: `+0.1440`
- `overblock_rate`: `+0.0042`

说明：LoRA 带来了显著收益，同时只增加了很小的 overblock。

### qwen_lora vs qwen_base

- `macro_f1`: `+0.0631`
- `unsafe_f1`: `+0.1039`
- `overblock_rate`: `-0.0274`

说明：`qwen_lora` 不只是更强，而且更稳；它比 `qwen_base` 更能抓 `unsafe`，同时更少误杀 `safe`。

## JSON 输出可解析性分析

本次 step 09 已额外加入与 step 08 解析逻辑一致的 JSON 可解析性分析，重点检查 `qwen_base` 和 `qwen_lora` 的原始 `prediction` 输出是否能被正确解析。

### 结果

| Model | Direct JSON Parse Rate | Direct Success / Total | extract_json_block Parse Rate | extract_json_block Success / Total | prediction_json Populated Rate |
| --- | --- | --- | --- | --- | --- |
| qwen_base | 0.6477 | 1107/1709 | 0.6477 | 1107/1709 | 0.6477 |
| qwen_lora | 0.9988 | 1707/1709 | 0.9988 | 1707/1709 | 0.9988 |

### 解读

- `qwen_base` 只有约 **64.8%** 的原始输出能被直接解析为 JSON。
- `qwen_lora` 约 **99.9%** 的原始输出能被直接解析为 JSON。
- 本次 `Direct JSON Parse Rate` 与 `extract_json_block Parse Rate` 完全一致，说明当前输出中基本不存在“外面包了一层说明文字，但里面仍有合法 JSON”的情况。
- 这意味着 LoRA 的收益不是单纯依赖后处理修复，而是**模型本身更稳定地直接生成目标 JSON 格式**。

## 结构化解释质量

| Model | raw_json_parse_success_rate | evidence_hit_rate | reason_label_consistency |
| --- | --- | --- | --- |
| baseline | 0.0000 | 0.0000 | 0.0000 |
| qwen_base | 0.6477 | 0.5290 | 0.6109 |
| qwen_lora | 0.9988 | 0.9105 | 0.9274 |

解读：

- `baseline` 基本不具备可比的结构化解释能力。
- `qwen_base` 已有一定结构化输出能力，但稳定性一般。
- `qwen_lora` 在 JSON 合法性、evidence 命中率和 reason-label 一致性上都明显最好，更接近可直接用于 judge 系统的目标形态。

## 最终结论

如果目标是训练一个输出 `label + reason + evidence` 的安全 judge 模型，那么本次 step 09 的结果表明：

> `qwen_lora` 是当前 Finetune-a-judge-model 项目里最可用的主模型。它同时提升了分类性能、降低了相对不必要的误杀、并显著增强了结构化 JSON 输出与解释字段质量。

## 相关文件

- 对比脚本：`scripts/09_compare_models.py`
- 单模型评估逻辑：`scripts/_eval_common.py`
- step 08 解析逻辑参考：`scripts/08_eval.py`
- 原始分析产物：`outputs/step09_compare/comparison_analysis.json`
- Markdown 分析产物：`outputs/step09_compare/comparison_analysis.md`
- checkpoint-654 对比分析产物：`outputs/eval_compare_checkpoint654/comparison_analysis.json`
- checkpoint-654 Markdown 分析产物：`outputs/eval_compare_checkpoint654/comparison_analysis.md`
- lr=5e-5 completion_only 对比分析产物：`outputs/eval_compare_lr_5e-5_completion_only/comparison_analysis.json`
- lr=3e-5 r32 对比分析产物：`outputs/eval_compare_lr_3e-5_r32/comparison_analysis.json`
