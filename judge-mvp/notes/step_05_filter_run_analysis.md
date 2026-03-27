# Step 05 运行记录：正式过滤结果与丢弃原因分析

## 1. 这份记录的定位

这份笔记覆盖旧版 step 05 记录，只保留当前这轮正式运行后仍然有效的信息。

对应脚本：
- [scripts/05_filter_pseudo_labels.py](../scripts/05_filter_pseudo_labels.py)

本轮 step 05 的输入，已经基于最新修复后的 step 04 产物：
- [data/interim/pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl)

也就是说，这份分析建立在以下前提上：
- step 04 已补齐到 `3000 / 3000`
- `teacher_output.label` 缺失已修复为 `0`
- 当前输入是 3000 条 balanced pseudo labels

---

## 2. 本轮正式运行命令与输出

正式运行命令：

```bash
uv run python scripts/05_filter_pseudo_labels.py
```

输入文件：
- [data/interim/pseudo_raw.jsonl](../data/interim/pseudo_raw.jsonl)

输出文件：
- 主过滤结果: [data/processed/pseudo_filtered.jsonl](../data/processed/pseudo_filtered.jsonl)
- label mismatch 保存文件: [data/processed/pseudo_label_mismatch.jsonl](../data/processed/pseudo_label_mismatch.jsonl)

本轮正式运行统计：

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
  "kept_label_counts": {
    "safe": 1206,
    "unsafe": 755
  },
  "label_mismatch_saved": 0
}
```

---

## 3. step 05 当前到底在过滤什么

当前 step 05 的核心过滤逻辑在：
- [scripts/05_filter_pseudo_labels.py:72-105](../scripts/05_filter_pseudo_labels.py#L72-L105)

它对每条 step 04 teacher 输出依次检查：

1. `teacher label` 是否等于原样本 `gold label`
2. `evidence` 是否存在且为非空列表
3. `evidence` 中每一项是否都能在 `question + response` 中做**严格子串匹配**
4. `reason` 长度是否位于 `[20, 400]`

只要某一步失败，该样本就不会进入主训练集。

其中最严格的一步是 evidence grounding：
- grounding 函数: [scripts/05_filter_pseudo_labels.py:48-55](../scripts/05_filter_pseudo_labels.py#L48-L55)
- 它要求 evidence 必须是原文本中的逐字片段，不接受意译或近似引用

---

## 4. 输入分布与过滤后分布

本轮 step 04 输入本来是严格平衡的：

- safe: `1500`
- unsafe: `1500`

过滤后保留下来的分布：

- safe kept: `1206`
- unsafe kept: `755`

过滤后被丢弃的分布：

- safe dropped: `294`
- unsafe dropped: `745`

按类别保留率：

- safe keep rate: `80.4%`
- unsafe keep rate: `50.33%`

这说明一个非常明确的现象：

> 当前 step 05 对 unsafe 样本更“苛刻”，unsafe 类样本更容易被过滤掉。

原因并不是 unsafe label 本身有问题，而是 unsafe 响应更容易触发：
- evidence 不够逐字 grounded
- reason 过长
- 解释过度展开

---

## 5. 本轮最重要的结论

本轮最重要的结论不是“保留了 1961 条”，而是：

> 当前 step 05 的主要损耗来源不是 label mismatch，而是 teacher evidence 无法满足严格的 verbatim grounding 要求。

具体看：

- `label_mismatch = 0`
- `evidence_not_grounded = 847`
- `reason_length_out_of_range = 185`
- `missing_evidence = 1`

这说明经过 step 04 的补跑与修复后：

- teacher 的主判断方向已经比较稳定
- gold / teacher label 已经对齐
- 当前最大问题转移到了 **explanation 质量控制**，尤其是 **evidence 是否真的逐字摘录**

---

## 6. 为什么样本不被保留：逐类分析

## 6.1 `evidence_not_grounded = 847`

这是最大的丢弃来源，占全部 dropped 的 **81.5%**。

这一类不是说 teacher 判断错了，而是说：

- teacher 给出了 evidence
- evidence 从语义上常常是对的
- 但 evidence 不是原文的严格逐字 span
- 因而过不了当前的 strict grounding 规则

这类样本的统计特征：
- count: `847`
- safe: `250`
- unsafe: `597`
- 平均 reason 长度: `350.82`
- 平均 evidence 条数: `3.57`

### 常见触发方式

#### A. teacher 把 evidence 写成意译或概括

例如 `train-53314`：
- label: `unsafe`
- teacher 判断本身合理
- 但 evidence 中包含：
  - `"Attention all seekers of truth!..."`
  - `To subtly hint at the unverified rumors and initiate open discussion...`

这里至少一部分 evidence 带引号或经过重新表述，无法被严格子串匹配命中。

#### B. teacher 对表格/结构化文本做了重新格式化

例如 `train-77836`：
- teacher 引用了表格行和表头
- 但格式、空格、排版和原始 response 不完全一致
- 因此 grounding 失败

#### C. teacher 用压缩引用代替逐字引用

例如一些 unsafe 样本中，teacher 会写成：
- 省略号 `...`
- 合并多段内容
- 去掉部分细节后概括为一条 evidence

这在语义上可能没问题，但不符合当前“精确可定位监督”的要求。

### 这一类的真正含义

`evidence_not_grounded` 不等于 teacher 大规模误判。

它更准确地意味着：

> teacher 的判断和解释通常能成立，但它没有稳定遵守“evidence 必须是 exact substring”的标注契约。

换句话说，当前 teacher 的 **reason 能力强于 evidence 精确摘录能力**。

---

## 6.2 `reason_length_out_of_range = 185`

这是第二大丢弃来源，占全部 dropped 的 **17.8%**。

当前长度阈值是：
- `min_reason_chars = 20`
- `max_reason_chars = 400`

见：
- [scripts/05_filter_pseudo_labels.py:63-64](../scripts/05_filter_pseudo_labels.py#L63-L64)
- [scripts/05_filter_pseudo_labels.py:98-100](../scripts/05_filter_pseudo_labels.py#L98-L100)

这类样本的统计特征：
- count: `185`
- safe: `37`
- unsafe: `148`
- 平均 reason 长度: `432.26`
- 最短: `401`
- 最长: `558`

也就是说，这一类几乎全是 **reason 太长**，而不是太短。

### 典型现象

例如：
- `train-28977`，`reason_len = 412`
- `train-58243`，`reason_len = 416`
- `train-31887`，`reason_len = 409`

这些样本的问题通常不是判断不对，而是：
- 解释太满
- 把多个判断层次都写进去了
- 生成风格更像长分析，而不是紧凑训练 target

### 这一类的真正含义

这说明当前 teacher 经常能给出高质量判断，但不够“收敛”。

对于 SFT 来说，过长 reason 会带来两个问题：
- supervision 不够稳定
- 训练目标风格不够统一

因此这 185 条被过滤是合理的。

---

## 6.3 `missing_evidence = 1`

这一类只有 1 条：
- `train-80422`

它的特点是：
- reason 很长，约 `604` 字符
- teacher 没给 evidence
- 因此直接被 drop

这不是系统性问题，只是极少数 teacher 输出结构不完整的尾部异常。

---

## 7. 关于已移除的 `refusal_style_reason` 规则

这一规则曾经尝试把带有 `I cannot assist`、`I'm sorry` 之类表达的 reason 过滤掉，出发点是避免把 teacher 自身拒答式输出当成 judge rationale。

但后续抽查发现，这类表达在当前数据里经常出现在：
- 原始 `response` 自身就是安全拒答
- teacher 在 `reason` 中对该 `response` 做归纳描述
- teacher 在 `evidence` 中逐字引用原始 `response`

也就是说，这更像是 **teacher 正在解释原始 response 为什么是 safe**，而不是 **teacher 自己在拒答**。

因此这条规则会误杀正常样本，现已从 step 05 中删除。当前 step 05 不再基于 refusal 风格文本单独 drop 样本。

---

## 8. 本轮没有出现的情况：`label_mismatch = 0`

这轮很重要的一点是：

> `pseudo_label_mismatch.jsonl` 最终是空的，`label_mismatch_saved = 0`。

这和旧版 step 05 记录已经不同了，旧结论现在应视为过时。

当前最新状态下：
- step 04 的补跑与 label 修复已经完成
- teacher label 与 gold label 没有出现批量不一致
- 当前 step 05 不再是“label disagreement 主导”的问题，而是“evidence / reason 质量控制主导”的问题

这说明前一轮真正暴露出的核心工程问题不是 label semantics 崩坏，而是：
- prompt 对 evidence verbatim 的约束不够强
- teacher 有时写得太长
- teacher 输出格式对 strict filter 不够友好

---

## 8. 对当前 step 05 结果的整体判断

### 8.1 主过滤结果是可用的

当前保留：
- `1961 / 3000`

对于 MVP 阶段来说，这是足够继续 step 06 / 07 的规模。

### 8.2 当前最大瓶颈已经非常明确

当前 step 05 最大瓶颈不是标签对齐，而是：

- evidence grounding 不够稳定
- unsafe 样本更容易因 evidence / reason 问题被过滤

### 8.3 这实际上在反向指导 step 04 prompt

step 05 的输出说明，下一轮 teacher prompt 调整方向应该重点针对：

1. **要求 evidence 必须是 exact span**
2. **禁止 paraphrase / summary / rewritten quote**
3. **尽量缩短 reason**

也就是说，step 05 已经不仅是过滤器，它实际上成了 step 04 prompt 调优的反馈信号。

---

## 9. 对后续工作的直接启发

你已经提出了一个很合理的后续方向：

> 扩展 step 04 的执行模式，允许基于 step 05 的 drop 结果，仅对会被过滤掉的样本重新跑 teacher。

从这次 step 05 结果看，这个方向非常值得做，原因是：

- 1039 条 dropped 里，大部分并不是语义错误
- 它们只是没有满足 strict filter 的格式/引用要求
- 如果换更强约束的 teacher prompt，仅对这些 drop 样本重跑，理论上有机会显著提高最终 kept 数

特别适合做二次补跑的，是：
- `evidence_not_grounded`
- `reason_length_out_of_range`

不太值得专门重跑的，是：
- `missing_evidence`（太少）

### 当前已经支持的 rerun workflow

当前脚本已经扩展为：

1. step 05 可直接导出 dropped 子集：
   - 默认输出 [data/interim/pseudo_dropped.jsonl](../data/interim/pseudo_dropped.jsonl)
   - 每行保留原始 step 04 字段，并新增：
     - `drop_reason`
     - `dropped_at_step`
     - `filter_config`

2. step 04 可直接把 dropped 子集当输入重跑：
   - 复用现有 `--input_path`
   - 推荐配合 `--sampling_strategy first_n`
   - 可加 `--rerun_tag` 区分这是否为一轮 drop-only rerun

推荐命令示例：

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

当前阶段建议：
- rerun 输出先写入独立文件
- 先观察新的 teacher prompt 是否真的改善 keep rate
- merge policy 之后再单独设计，而不是现在就自动覆盖主文件

---

## 10. 一句话总结

这轮 step 05 的正式结果说明：

> 当前过滤链路已经稳定，但瓶颈非常集中：3000 条 step 04 teacher 输出中，真正阻止样本进入主训练集的主要原因不是 label 错，而是 teacher 没有稳定产出“可逐字定位的 evidence”和“长度受控的 concise reason”；因此下一步最值得做的不是放宽 step 05，而是让 step 04 支持针对 step 05 drop 样本的定向重跑。
