# Step 09 模型对比分析总结

## 本次运行概览

- 参考集：`data/processed/test.jsonl`
- 样本数：1709
- 标签分布：`safe=1425`，`unsafe=284`
- 对比对象：`baseline`、`qwen_base`、`qwen_lora`

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

> `qwen_lora` 是当前 judge-mvp 中最可用的主模型。它同时提升了分类性能、降低了相对不必要的误杀、并显著增强了结构化 JSON 输出与解释字段质量。

## 相关文件

- 对比脚本：`scripts/09_compare_models.py`
- 单模型评估逻辑：`scripts/_eval_common.py`
- step 08 解析逻辑参考：`scripts/08_eval.py`
- 原始分析产物：`outputs/step09_compare/comparison_analysis.json`
- Markdown 分析产物：`outputs/step09_compare/comparison_analysis.md`
