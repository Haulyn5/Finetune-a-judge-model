# Step 07 运行记录：SFT LoRA 训练分析（2026-03-31）

## 1. 本轮记录的定位

这份笔记记录 2026-03-31 这次基于扩充后 SFT 数据的 step 07 主训练运行，包括：

- 训练输入与超参数
- checkpoint 保存情况
- dev 集上的关键评估结果
- 最后一个 epoch 出现异常的具体表现
- 为什么这次 run 不能直接把最后 checkpoint 当作 step 08 的评估对象
- 为后续重新训练时启用 best-checkpoint selection 提供依据

对应脚本：
- [scripts/07_train_sft_lora.py](../scripts/07_train_sft_lora.py)
- [scripts/08_eval.py](../scripts/08_eval.py)

---

## 2. 本轮训练使用的数据与输出目录

本轮训练使用的是扩充后的 15000 teacher 数据经过 step 05/06 处理后的 SFT 数据：

- train: [data/processed/sft_train_15000.jsonl](../data/processed/sft_train_15000.jsonl)
- dev: [data/processed/sft_dev_15000.jsonl](../data/processed/sft_dev_15000.jsonl)

样本规模：
- train examples: `10463`
- eval examples: `1163`

本轮输出目录：
- [outputs/sft_lora_15000/](../outputs/sft_lora_15000/)

本轮实际保存出的关键产物：
- 最终导出的 adapter: [outputs/sft_lora_15000/adapter_model.safetensors](../outputs/sft_lora_15000/adapter_model.safetensors)
- 中间 checkpoint:
  - [outputs/sft_lora_15000/checkpoint-654/](../outputs/sft_lora_15000/checkpoint-654/)
  - [outputs/sft_lora_15000/checkpoint-981/](../outputs/sft_lora_15000/checkpoint-981/)
- TensorBoard runs:
  - [outputs/sft_lora_15000/runs/](../outputs/sft_lora_15000/runs/)

---

## 3. 本轮训练配置

本轮训练使用的是 Qwen3.5-4B 主路径，LoRA 微调，关键参数如下：

- base model: `/root/project/PretrainedModels/Qwen/Qwen3.5-4B`
- run name: `step07_sft_lora_15000`
- output dir: `outputs/sft_lora_15000`
- num_train_epochs: `3`
- per_device_train_batch_size: `4`
- per_device_eval_batch_size: `2`
- gradient_accumulation_steps: `8`
- learning_rate: `1e-4`
- weight_decay: `0.01`
- max_seq_length: `2048`
- logging_steps: `10`
- save_total_limit: `2`
- precision: `bf16`
- load_in_4bit: `False`
- gradient_checkpointing: `True`
- eval strategy: `epoch`
- save strategy: `epoch`
- report_to: `tensorboard`

LoRA 配置：
- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `up_proj`
  - `down_proj`
  - `gate_proj`

---

## 4. dev 集表现：前两个 epoch 正常，第三个 epoch 崩坏

从 [outputs/sft_lora_15000/checkpoint-981/trainer_state.json](../outputs/sft_lora_15000/checkpoint-981/trainer_state.json) 中提取到的 eval 历史如下：

```json
[
  {
    "epoch": 1.0,
    "eval_loss": 0.9795956611633301,
    "eval_mean_token_accuracy": 0.7645334911715124,
    "step": 327
  },
  {
    "epoch": 2.0,
    "eval_loss": 0.9546372890472412,
    "eval_mean_token_accuracy": 0.7690084074268636,
    "step": 654
  },
  {
    "epoch": 3.0,
    "eval_loss": NaN,
    "eval_mean_token_accuracy": 0.0006301803443852102,
    "step": 981
  }
]
```

### 4.1 epoch 1

- step: `327`
- eval_loss: `0.9796`
- eval_mean_token_accuracy: `0.7645`

这是一个正常起点。

### 4.2 epoch 2

- step: `654`
- eval_loss: `0.9546`
- eval_mean_token_accuracy: `0.7690`

相比 epoch 1：
- `eval_loss` 继续下降
- `eval_mean_token_accuracy` 略有提升

也就是说，**第 2 个 epoch 是本轮训练中最好的阶段**。

### 4.3 epoch 3

- step: `981`
- eval_loss: `NaN`
- eval_mean_token_accuracy: `0.00063`

这一轮出现了明显的训练异常：
- dev loss 直接变成 `NaN`
- token accuracy 几乎掉到 `0`

这与用户在 TensorBoard 中看到的：
- 最后一个 epoch 的某个 step 之后 loss spike
- accuracy 显著下降并无法恢复

是完全一致的。

---

## 5. 训练日志末尾的异常征兆

在最后阶段的 `log_history` 里还能看到：

- `entropy = NaN`
- `grad_norm = NaN`
- `loss = 0.0`
- `mean_token_accuracy` 接近 `0`

例如 step 920~980 之间已经持续出现：
- `entropy: NaN`
- `grad_norm: NaN`
- `loss: 0.0`
- `mean_token_accuracy` 仅约 `0.0005 ~ 0.0009`

这说明最后阶段不是普通波动，而是训练状态已经明显异常。

---

## 6. 为什么这次 run 不能直接用最后 checkpoint 做 step 08

本轮训练结束时，由于 당시脚本版本还没有 best-checkpoint selection：

- `best_metric = null`
- `best_model_checkpoint = null`

也就是说：
- 这次旧 run 没有自动记录“最佳 checkpoint”
- 根目录导出的 adapter 不能默认视为“最佳 dev 模型”
- 最后一个 checkpoint（`checkpoint-981`）尤其不能直接拿去做正式评估

从 dev 曲线来看，这次 run 中最合理的 checkpoint 实际上应当是：

- [outputs/sft_lora_15000/checkpoint-654/](../outputs/sft_lora_15000/checkpoint-654/)

因为它对应：
- epoch 2
- 最低的有限 `eval_loss`
- 最好的 `eval_mean_token_accuracy`

---

## 7. 对异常 checkpoint-981 做的两样本推理观察

为了理解“训练崩坏后模型会怎么表现”，本轮额外对最后 checkpoint 做了一个 2 样本的小规模推理检查。

使用 checkpoint：
- [outputs/sft_lora_15000/checkpoint-981/](../outputs/sft_lora_15000/checkpoint-981/)

使用样本文件：
- [data/interim/test_two_samples_checkpoint981.jsonl](../data/interim/test_two_samples_checkpoint981.jsonl)

生成输出：
- [outputs/debug_checkpoint981_two_samples.jsonl](../outputs/debug_checkpoint981_two_samples.jsonl)

评估结果：
- [outputs/debug_checkpoint981_two_samples_metrics.json](../outputs/debug_checkpoint981_two_samples_metrics.json)

### 7.1 观察到的直接现象

两条样本的生成都退化成了大量重复的感叹号：

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

并且：
- `prediction_json = null`
- `reason = ""`
- `evidence = []`

也就是说，**结构化输出能力已经整体崩坏**，不只是 label 判别轻微偏移。

### 7.2 标签表现

由于 step 08 的 fallback 逻辑在无法解析有效 JSON 时会默认回落到 `safe`，因此这 2 条样本表现为：

- safe 样本 -> 预测 `safe`
- unsafe 样本 -> 也预测 `safe`

对应评估：

```json
{
  "accuracy": 0.5,
  "macro_f1": 0.3333333333333333,
  "unsafe_precision": 0.0,
  "unsafe_recall": 0.0,
  "unsafe_f1": 0.0,
  "raw_json_parse_success_rate": 0.0,
  "evidence_hit_rate": 0.0,
  "reason_label_consistency": 0.0
}
```

### 7.3 结论

`checkpoint-981` 的问题不是“泛化稍差”，而是：

- 生成头已经进入异常重复 token 状态
- JSON 结构无法维持
- unsafe 判别能力实际上塌缩
- 不适合作为 step 08 的正式评估对象

---

## 8. 本轮最重要的结论

这次 step 07 的最重要结论不是“3 个 epoch 跑完了”，而是：

> 本轮训练在前两个 epoch 是正常收敛的，但第 3 个 epoch 后半段发生了明显崩坏；从 dev 指标和异常 checkpoint 的实际生成结果看，问题已经不是轻微过拟合，而是生成行为整体失控，因此本轮正式可用的最佳模型应视为 `checkpoint-654`，而不是最后的 `checkpoint-981`。

---

## 9. 对后续 step 08 的直接建议

在这次旧 run 上，如果现在立刻要做 step 08 正式评估，建议优先使用：

- [outputs/sft_lora_15000/checkpoint-654/](../outputs/sft_lora_15000/checkpoint-654/)

而不是：
- [outputs/sft_lora_15000/checkpoint-981/](../outputs/sft_lora_15000/checkpoint-981/)
- 也不要默认使用根目录 [outputs/sft_lora_15000/](../outputs/sft_lora_15000/)，因为这次 run 当时还没有 best-checkpoint 自动回载逻辑。

后续重新训练时，已经改好的 [scripts/07_train_sft_lora.py](../scripts/07_train_sft_lora.py) 会：

- 自动根据 dev `eval_loss` 选择最佳 checkpoint
- 在训练结束时输出 `best_model_checkpoint`
- 将最佳模型导出到输出根目录
- 生成 `training_summary.json`

这样下一轮 step 08 就可以直接用新的输出根目录做评估，而不必再人工判断哪个 checkpoint 最好。
