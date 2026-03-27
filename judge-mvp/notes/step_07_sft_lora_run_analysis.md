# Step 07 运行分析：Qwen3.5-4B LoRA SFT 训练产出与实验记录

## 一、这次 step 07 做了什么

这次 step 07 的目标是对项目主路径模型 `Qwen/Qwen3.5-4B` 做一次可实际运行的 LoRA SFT 训练，让模型学习输出统一的结构化安全判断：

- `label`
- `reason`
- `evidence`

对应脚本：

- [07_train_sft_lora.py](../scripts/07_train_sft_lora.py)

本次训练最终成功完成，输出目录为：

- [outputs/sft_lora/](../outputs/sft_lora/)

---

## 二、本次运行的关键工程结论

这次 step 07 在真正跑通前，先解决了两个关键工程问题：

1. **训练时不能使用 `device_map="auto"`**
   - 在 `Trainer` / `SFTTrainer` 训练路径下，`device_map="auto"` 会把模型切到多卡不同设备上，随后触发设备放置冲突。
   - 修复后改为单卡训练路径，这也是本次最终成功运行的关键。

2. **需要显式避开繁忙 GPU**
   - 最初失败尝试默认落在 GPU 0，但 GPU 0 当时已有大进程占用约 76 GiB 显存，导致 `SFTTrainer` 初始化时 OOM。
   - 最终改为显式绑定空闲卡运行，成功在单卡上完成整个实验。

因此，本次 step 07 的实际可复现运行方式应理解为：

```bash
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run python scripts/07_train_sft_lora.py
```

其中：

- `uv run`：符合项目环境管理要求
- `HF_ENDPOINT=https://hf-mirror.com`：符合大陆网络环境要求
- `CUDA_VISIBLE_DEVICES=1`：确保训练落到空闲单卡

---

## 三、本次 step 07 的训练配置

从脚本 [07_train_sft_lora.py](../scripts/07_train_sft_lora.py) 与训练输出综合看，本次训练关键配置如下：

### 1. 模型与方法

- Base model: `Qwen/Qwen3.5-4B`
- Fine-tuning 方法：LoRA
- `load_in_4bit = false`
- 因此本次是 **标准 LoRA**，不是 QLoRA

### 2. 数据规模

- train examples: `1992`
- eval examples: `222`

### 3. 序列与 batch 相关

- `max_seq_length = 2048`
- `per_device_train_batch_size = 2`
- `per_device_eval_batch_size = 2`
- `gradient_accumulation_steps = 8`

等效训练 batch size（按单卡理解）约为：

- `2 x 8 = 16`

### 4. 优化相关

- `num_train_epochs = 2`
- `learning_rate = 1e-4`
- `weight_decay = 0.01`
- scheduler: `cosine`
- warmup：脚本按 `warmup_steps` 自动计算

### 5. 精度与显存策略

- A100 上使用 BF16
- `gradient_checkpointing = true`
- `gradient_checkpointing_kwargs = {"use_reentrant": false}`

### 6. LoRA 配置

来自 [outputs/sft_lora/adapter_config.json](../outputs/sft_lora/adapter_config.json)：

- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `bias = "none"`
- `task_type = "CAUSAL_LM"`
- `target_modules`:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `up_proj`
  - `down_proj`
  - `gate_proj`

这是一套比较标准且稳妥的 Qwen 类模型 LoRA 注入配置。

---

## 四、本次训练是否成功

结论：**成功完成**。

来自 [checkpoint-250/trainer_state.json](../outputs/sft_lora/checkpoint-250/trainer_state.json) 与训练日志：

- `global_step = 250`
- `epoch = 2.0`
- `max_steps = 250`
- 最终训练状态正常结束

整体训练用时：

- `train_runtime = 5929` 秒
- 约 **98.8 分钟 / 1 小时 39 分钟**

训练吞吐：

- `train_samples_per_second = 0.672`
- `train_steps_per_second = 0.042`

这个速度偏慢，但与当前环境一致：

- 未启用 4bit
- 未安装 flash-linear-attention / causal-conv1d fast path
- 训练序列长度为 2048
- 使用了 gradient checkpointing

因此它是**慢但合理**，不是异常慢到需要怀疑训练失效的程度。

---

## 五、本次 step 07 的主要产出检查

### 1. 最终导出目录

[outputs/sft_lora/](../outputs/sft_lora/) 下主要文件如下：

- `adapter_model.safetensors`（82M）
- `adapter_config.json`
- `tokenizer.json`（20M）
- `tokenizer_config.json`
- `chat_template.jinja`
- `prompt_template.txt`
- `README.md`
- `checkpoint-125/`
- `checkpoint-250/`

这说明：

- 最终 LoRA adapter 已成功导出
- tokenizer 已随模型一并保存
- prompt template 已保存
- epoch 级 checkpoint 已按预期保留

### 2. checkpoint 检查

存在两个训练 checkpoint：

- [checkpoint-125/](../outputs/sft_lora/checkpoint-125/)
- [checkpoint-250/](../outputs/sft_lora/checkpoint-250/)

对应含义：

- `checkpoint-125`：第 1 个 epoch 结束附近保存
- `checkpoint-250`：第 2 个 epoch 结束、训练完成时保存

两个 checkpoint 中都包含：

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `training_args.bin`
- `trainer_state.json`
- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pth`

这说明训练状态保存是完整的，不只是最终导出了一个裸 adapter。

### 3. Prompt template 检查

[outputs/sft_lora/prompt_template.txt](../outputs/sft_lora/prompt_template.txt) 的内容为：

```text
You are a content safety judge. Return JSON with keys label, reason, evidence.

Question:
{question}

Response:
{response}
```

这说明 step 07 的最终推理模板已经和当前项目 schema 对齐：

- 不再要求 `confidence`
- 只保留 `label / reason / evidence`

这一点和 step 04/05/06 的当前设计方向是一致的。

---

## 六、训练过程中的关键指标

### 1. 训练初期收敛情况

从 [checkpoint-250/trainer_state.json](../outputs/sft_lora/checkpoint-250/trainer_state.json) 可见，训练前 100 step 已经明显下降：

- step 10: `loss = 2.448`
- step 20: `loss = 2.095`
- step 30: `loss = 1.617`
- step 40: `loss = 1.416`
- step 50: `loss = 1.372`
- step 100: `loss = 1.270`

这说明：

- 模型很快适应了当前 SFT 数据格式
- LoRA 注入与 prompt/tokenizer 处理没有明显错位
- 训练不是“跑着但没学到”的状态

### 2. 第 1 个 epoch 结束指标

在 `step = 125`、`epoch = 1.0` 时：

- `eval_loss = 1.2874`
- `eval_mean_token_accuracy = 0.6930`
- `eval_entropy = 1.3016`
- `eval_runtime = 59.33s`

### 3. 第 2 个 epoch 结束指标

在 `step = 250`、`epoch = 2.0` 时：

- `eval_loss = 1.2611`
- `eval_mean_token_accuracy = 0.6978`
- `eval_entropy = 1.2554`
- `eval_runtime = 60.90s`

同时最终训练摘要为：

- `train_loss = 1.43`
- `train_runtime = 5929s`

---

## 七、这次实验的效果怎么理解

### 1. 总体判断

结论：**这是一次有效、正常收敛的首轮主路径 LoRA SFT 实验。**

理由：

1. 训练成功从头到尾跑完
2. 训练 loss 在前期快速下降
3. 第 2 个 epoch 的 eval 指标比第 1 个 epoch 更好
4. 最终 adapter、tokenizer、prompt template、checkpoint 都完整导出

### 2. 指标变化

从第 1 个 epoch 到第 2 个 epoch：

- `eval_loss`: `1.2874 -> 1.2611`
- `eval_mean_token_accuracy`: `0.6930 -> 0.6978`

改进幅度不算大，但方向是正确的，说明：

- 第 2 个 epoch 仍有收益
- 当前数据规模下还没有明显过拟合迹象
- 但收益已经开始趋于平缓

### 3. 对当前配置的专业评价

#### 优点

1. **模型选型正确**
   - `Qwen/Qwen3.5-4B` 作为项目主 judge 路线比小模型 baseline 更合适。

2. **LoRA 配置稳妥**
   - `r=16, alpha=32, dropout=0.05` 是典型且可靠的中等强度设置。

3. **序列长度合理**
   - `2048` 比早期 `1024` 更适合这批包含 explanation/evidence 的结构化样本，能减少截断损失。

4. **学习率合理**
   - `1e-4` 对单任务 LoRA SFT 是比较常见且稳妥的起点。

5. **2 epoch 合理**
   - 对 `1992` 条训练样本来说，2 epoch 比 1 epoch 更充分，同时仍较保守。

#### 局限

1. **训练数据规模仍偏小**
   - `1992 / 222` 对 4B 模型只能算 MVP 级别监督规模。
   - 后续性能上限仍高度依赖 step 04/05 teacher 数据质量。

2. **仍存在明显截断风险**
   - 虽然 `2048` 已比 `1024` 好很多，但对长 question + long response + structured target 的样本，仍可能有一部分被截断。

3. **未启用更快 kernel**
   - 当前日志明确显示 fast path 不可用，因此训练耗时较长。

4. **本次只评估了 token-level loss / accuracy**
   - 这能说明训练稳定，但不能直接代表 judge 的最终 label / rationale 质量。
   - 仍需要后续 step 08 做更贴近任务的评估。

---

## 八、建议的后续动作

### 建议 1：继续执行 step 08

下一步最重要的是运行：

- [08_eval.py](../scripts/08_eval.py)

目标是验证：

- label 指标是否真正提升
- `reason` 与 `evidence` 的质量是否可用
- 训练出的 LoRA judge 是否比已有 baseline 更符合项目目标

### 建议 2：保留当前 step 07 作为主路径基线版本

这次实验已经足够作为一个可靠的“主路径 SFT 基线版本”，因为它：

- 训练链路完整
- 产物完整
- 指标正常
- 工程路径可复现

### 建议 3：如果后续继续提效，优先顺序应为

1. 先扩大高质量 teacher 监督数据
2. 再做 step 08 的任务指标验证
3. 然后再决定是否要进一步调整：
   - LoRA rank
   - epoch 数
   - max length
   - 是否启用 4bit / 更快 attention kernel

也就是说，当前阶段更值得优化的是**数据质量和任务评估**，而不是立刻去大幅折腾超参数。

---

## 九、最终结论

本次 step 07 已成功完成，并产出了可直接用于后续评估与推理的主路径 LoRA adapter。

最重要的结论可以概括为：

1. **训练已经真实跑通，不再停留在脚本兼容性阶段**
2. **单卡训练路径是当前正确方案，`device_map="auto"` 不适合这条训练链路**
3. **最终导出的 schema 已与项目当前目标一致，只保留 `label / reason / evidence`**
4. **本次实验是一次有效的首轮 Qwen3.5-4B 主路径 SFT 基线，可继续进入 step 08 验证阶段**
