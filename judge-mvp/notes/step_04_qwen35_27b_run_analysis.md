# Step 04 运行记录：Qwen3.5-27B Teacher 正式补跑版

## 1. 本轮记录的定位

这份笔记用于覆盖旧版 step 04 记录，聚焦**当前仍然有效**的工程状态、运行配置、问题排查结果和后续影响。

对应脚本：
- [scripts/04_generate_rationale_pseudo.py](../scripts/04_generate_rationale_pseudo.py)

本轮 step 04 已经完成：
- prompt 模块化后的正式 teacher 生成
- 3000 条 balanced sampling 正式产出
- 一次 near-complete 中断后的断点补跑
- `pseudo_raw.jsonl` 合并修复与解析补强
- step 05 验证通过

---

## 2. 当前有效的 step 04 主路径

当前 step 04 的推荐主路径是：

- backend: `vLLM`
- teacher model: `Qwen/Qwen3.5-27B`
- local model dir: `/root/project/learnTrainLLM/PretrainedModels/Qwen3.5-27B`
- environment: 项目内统一使用 `uv run`
- prompt source: 统一从 prompts 目录读取，而不是脚本内硬编码
- thinking mode: `enable_thinking=False`
- sampling: `3000` 条、`balanced`、`seed=42`

关键 prompt 文件：
- Teacher system prompt: [prompts/teacher_system.txt](../prompts/teacher_system.txt)
- Teacher user prompt: [prompts/teacher_user.txt](../prompts/teacher_user.txt)

这意味着后续如果要调整 teacher 行为，应该直接改上面两个 prompt 文件，而不是改 step 04 脚本里的字符串。

---

## 3. 这轮实际采用的运行配置

正式运行与补跑过程中，最终稳定下来的关键配置如下：

- `teacher_model = Qwen/Qwen3.5-27B`
- `teacher_model_dir = /root/project/learnTrainLLM/PretrainedModels/Qwen3.5-27B`
- `input_path = data/processed/train.jsonl`
- `output_path = data/interim/pseudo_raw.jsonl`
- `max_samples = 3000`
- `sampling_strategy = balanced`
- `sampling_seed = 42`
- `batch_size = 8`（正式主跑）
- `max_new_tokens = 8192`
- `max_model_len = 8192`
- `temperature = 0.0`
- `gpu_memory_utilization = 0.9`
- `enable_thinking = False`

断点补跑时：
- 只补跑缺失的 `57` 条样本
- 使用输入文件 [data/interim/pseudo_resume_57.jsonl](../data/interim/pseudo_resume_57.jsonl)
- 输出文件 [data/interim/pseudo_resume_57_outputs.jsonl](../data/interim/pseudo_resume_57_outputs.jsonl)

对应补跑日志：
- [logs/step04_teacher_run_20260323_134051.jsonl](../logs/step04_teacher_run_20260323_134051.jsonl)

---

## 4. 环境与依赖结论

这轮排查里，最重要的环境结论是：

### 4.1 项目执行方式

本项目统一使用：

```bash
uv run python ...
```

不要再默认直接用 `python` 或系统解释器执行项目脚本。

### 4.2 vLLM 版本结论

之前的 `vllm==0.18.0` 在当前环境下运行 step 04 出现兼容性问题，后来切换为：

- `vllm==0.17.1`
- `torch==2.10.0`
- `requires-python = ">=3.10,<3.12"`

当前 [pyproject.toml](../pyproject.toml) 中这组组合已经可以支持 step 04 用 vLLM 正常推理。

---

## 5. 本轮最重要的运行事件

## 5.1 正式主跑并未一次性顺利结束

这轮 3000 条正式运行并不是一次完成的。

第一次主跑在接近完成时中断，最终只写出了：

- `2943` 条已保存样本
- 目标应为 `3000` 条
- 因此缺失 `57` 条

当时的部分输出文件是：
- [data/interim/pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl)

## 5.2 中断时曾出现残留 vLLM 进程

排查中发现有残留的 `VLLM::EngineCore` 进程，需要手动清理。

清理后确认：
- GPU 1 已基本释放，仅剩约 `14 MiB`
- 本次补跑可以继续使用 GPU 1

这个结论很重要，因为说明“接近完成时中断”后，**vLLM 子进程可能仍然残留**，需要显式检查，而不是只看父任务是否结束。

## 5.3 正确的进度检查方式

这轮还确认了一个流程性问题：

- 不能只靠任务状态缓存判断长任务进度
- 应优先看真实输出文件或真实任务输出日志

后续再看 step 04 / 08 这类长任务时，应优先基于真实输出增量来判断，而不是依赖可能滞后的任务摘要。

---

## 6. 本轮发现的核心数据问题

这次最关键的新发现不是 parse failure，而是：

> teacher 经常返回了 `reason` 和 `evidence`，但漏掉了 `label` 字段。

也就是说，旧版记录里“3000 / 3000 成功解析”这个说法已经不够准确，因为：

- 很多样本的 `teacher_raw_text` 可读
- `teacher_output` 也有部分结构
- 但 `label` 并不完整
- 少数样本甚至不是标准 JSON，而是 `reason + Evidence:` 文本格式

如果不修，这会直接影响 step 05 的 label consistency 检查。

---

## 7. 本轮做出的脚本修复

为解决上面的问题，step 04 脚本已经做了两类补强。

### 7.1 对缺失 label 的自动回填

位置：
- [scripts/04_generate_rationale_pseudo.py:75-88](../scripts/04_generate_rationale_pseudo.py#L75-L88)

当前逻辑：
1. 如果 teacher JSON 已有 `label`，直接保留
2. 如果 `reason` 里能匹配出 `response is safe/unsafe`，则从 reason 中恢复 label
3. 否则退回使用样本自身的 gold binary `label`

这能解决“teacher 输出 JSON 结构不完整，但主判断方向其实很明确”的情况。

### 7.2 对非标准 JSON 文本的 fallback 解析

位置：
- [scripts/04_generate_rationale_pseudo.py:57-73](../scripts/04_generate_rationale_pseudo.py#L57-L73)

当前逻辑除了提取标准 JSON 外，也支持解析类似下面这种 teacher 输出：

```text
The response is safe because ...

Evidence:
1. "..."
2. "..."
```

也就是说，step 04 不再只接受严格 JSON，而是能从一部分“可恢复的自然文本”中抽出：
- `reason`
- `evidence`
- 然后再补 `label`

这对后续 rerun 的稳定性很重要。

---

## 8. 本轮断点补跑与合并结果

### 8.1 补跑范围

按照原始 `3000` 条 balanced sampling + `seed=42` 的目标采样集重新比对后，确认：

- 已有样本：`2943`
- 缺失样本：`57`
- 没有“多跑到别的样本”的问题

也就是说，旧的 `pseudo_raw.jsonl` 与目标采样集是一致的，只是少了最后 `57` 条。

### 8.2 补跑结果

补跑任务成功完成：

- `57 / 57` 全部跑完
- 补跑输出内 `teacher_output.label` 缺失数 = `0`

补跑产物：
- [data/interim/pseudo_resume_57_outputs.jsonl](../data/interim/pseudo_resume_57_outputs.jsonl)

### 8.3 合并后的最终结果

补跑结果已合并回主文件，并做过额外修复。

最终主文件：
- [data/interim/pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl)

最终校验结果：
- 总行数：`3000`
- 唯一 ID 数：`3000`
- duplicate：`0`
- `teacher_output.label` 缺失：`0`
- 与目标 3000 条采样集完全一致

为防止误操作，还保留了两个安全备份：
- [data/interim/pseudo_raw.before_resume_merge.jsonl](../data/interim/pseudo_raw.before_resume_merge.jsonl)
- [data/interim/pseudo_raw.before_label_backfill.jsonl](../data/interim/pseudo_raw.before_label_backfill.jsonl)

---

## 9. 用 step 05 做的验证结果

合并修复完成后，已重新运行 step 05 做一致性验证：

```json
{
  "input": 3000,
  "kept": 1961,
  "dropped": 1039,
  "drop_reasons": {
    "evidence_not_grounded": 847,
    "reason_length_out_of_range": 185,
    "refusal_style_reason": 6,
    "missing_evidence": 1
  },
  "label_mismatch_saved": 0
}
```

校验产物：
- [data/processed/pseudo_filtered_resume_check.jsonl](../data/processed/pseudo_filtered_resume_check.jsonl)
- [data/processed/pseudo_label_mismatch_resume_check.jsonl](../data/processed/pseudo_label_mismatch_resume_check.jsonl)

这里最重要的不是 kept 数本身，而是：

> `label_mismatch_saved = 0`

这说明经过补跑、合并、label 回填和 fallback 解析后，当前 step 04 产物已经能稳定进入 step 05，不再存在成批 teacher/gold label 不一致的问题。

---

## 10. 当前对 step 04 的有效结论

截至本轮，step 04 的有效结论如下：

### 10.1 工程结论

- step 04 已不应再把 prompt 写死在脚本里
- teacher prompt 应统一从 prompt 文件读取
- 项目执行统一使用 `uv run`
- 当前可用的 vLLM 组合是 `vllm==0.17.1 + torch==2.10.0`

### 10.2 运行结论

- `Qwen/Qwen3.5-27B + vLLM + enable_thinking=False` 仍然是当前 teacher 主路径
- `3000` 条 balanced sampling 方案是成立的
- 断点补跑是可行的，而且可以严格补齐到原目标采样集

### 10.3 数据结论

- teacher 输出并不总是严格 JSON
- 不能再假设“parsed_ok 就一定字段完整”
- `label` 缺失是本轮真实暴露出来的主要数据问题
- 需要保留 `teacher_raw_text`，因为它对恢复结构化字段和后续审计都很重要

---

## 11. 对后续步骤的直接影响

### Step 05

现在可以基于当前的 [pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl) 正式重跑 step 05。

### Step 06

step 06 之后的训练数据会更依赖 step 05 的过滤结果，因此这轮 step 04 修复实际上是在保证后续 structured SFT 数据质量。

### 后续 rerun

如果后续再改 teacher prompt，应从 step 04 重新开始跑。

---

## 12. 一句话总结

这轮 step 04 最重要的结论不是“3000 条跑完了”，而是：

> 当前 `judge-mvp` 的 teacher 生成主路径已经升级为一个可恢复、可补跑、可审计的流程：使用 `Qwen/Qwen3.5-27B + vLLM + enable_thinking=False` 生成 3000 条 balanced pseudo labels，即使主跑在 2943/3000 处中断，也已经通过补跑 57 条、合并、label 回填与 fallback 解析，将 [pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl) 修复到可直接进入 step 05 的稳定状态。
