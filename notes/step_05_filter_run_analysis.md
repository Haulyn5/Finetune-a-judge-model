# Step 05 运行记录：15000 条 teacher 数据过滤结果（当前主版本）

## 1. 本文定位

这份笔记以**最新完成且当前有效**的 step 05 结果为主，记录基于 15000 条 teacher 数据的正式过滤输出、关键统计、主要丢弃原因，以及对后续 step 06 / rerun 的直接意义。

旧的 3000 条版本记录仍然有参考价值，但现在已经降级为历史背景。当前后续训练与分析应优先基于：
- [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl)

---

## 2. 本轮输入与输出

本轮 step 05 使用的输入是：
- [pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

这是 step 04 扩充后的主 teacher 文件，规模为：
- `15000` 条
- 标签目标来自 step 04 扩充结果：`safe=7500`、`unsafe=7500`

为避免覆盖旧版 step 05 结果，这次输出写到了新的文件名：

- 主过滤结果： [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl)
- label mismatch 子集： [pseudo_label_mismatch_15000.jsonl](../data/processed/pseudo_label_mismatch_15000.jsonl)
- 全部 dropped 子集： [pseudo_dropped_15000.jsonl](../data/interim/pseudo_dropped_15000.jsonl)

这些文件都已经存在，可继续直接供后续步骤使用。

---

## 3. 本轮执行命令

本轮实际执行命令是：

```bash
uv run python scripts/05_filter_pseudo_labels.py \
  --input_path data/interim/pseudo_raw_15000.jsonl \
  --output_path data/processed/pseudo_filtered_15000.jsonl \
  --label_mismatch_output_path data/processed/pseudo_label_mismatch_15000.jsonl \
  --dropped_output_path data/interim/pseudo_dropped_15000.jsonl
```

对应脚本：
- [scripts/05_filter_pseudo_labels.py](../scripts/05_filter_pseudo_labels.py)

---

## 4. step 05 当前到底在过滤什么

当前 step 05 的核心逻辑没有变，仍然是对每条 step 04 teacher 输出做结构与一致性过滤。

关键逻辑位置：
- [scripts/05_filter_pseudo_labels.py:49-74](../scripts/05_filter_pseudo_labels.py#L49-L74)
- [scripts/05_filter_pseudo_labels.py:91-167](../scripts/05_filter_pseudo_labels.py#L91-L167)

每条样本会依次检查：

1. `teacher_output.label` 是否与原样本 `label` 一致
2. `evidence` 是否存在且至少 1 条
3. 每条 `evidence` 是否都能在 `question + response` 中做**严格子串匹配**
4. `reason` 长度是否位于 `[20, 400]`

只要任意一步失败，这条样本就不会进入最终的主训练集。

其中最严格、也是本轮最大的过滤来源，仍然是：
- `evidence_not_grounded`

因为这里要求 evidence 是原文中的**逐字片段**，不接受意译、概括、重写后的引用。

---

## 5. 本轮最终统计结果

本轮 step 05 输出统计如下：

```json
{
  "input": 15000,
  "kept": 11626,
  "dropped": 3374,
  "drop_reasons": {
    "evidence_not_grounded": 2160,
    "reason_length_out_of_range": 841,
    "label_mismatch": 373
  },
  "kept_label_counts": {
    "safe": 6489,
    "unsafe": 5137
  },
  "label_mismatch_saved": 373
}
```

额外行数校验：
- [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl): `11626` 行
- [pseudo_label_mismatch_15000.jsonl](../data/processed/pseudo_label_mismatch_15000.jsonl): `373` 行
- [pseudo_dropped_15000.jsonl](../data/interim/pseudo_dropped_15000.jsonl): `3374` 行

### 核心结论

- 总输入：`15000`
- 保留：`11626`
- 丢弃：`3374`
- 整体保留率：**77.51%**

这说明：
- 15000 条 step 04 teacher 数据中，已有相当大比例满足当前 strict filter
- 对 MVP 训练来说，这已经是一个明显更强的数据基座

---

## 6. 输入分布与过滤后分布

step 04 扩充完成后，输入是严格平衡的：
- safe: `7500`
- unsafe: `7500`

### 过滤后保留分布

- safe kept: `6489`
- unsafe kept: `5137`

### dropped 分布

从 `pseudo_dropped_15000.jsonl` 统计得到：
- safe dropped: `1011`
- unsafe dropped: `2363`

### 按类别保留率

- safe keep rate: `6489 / 7500 = 86.52%`
- unsafe keep rate: `5137 / 7500 = 68.49%`

这说明一个非常明确的现象：

> 即使 step 04 输入已经做到 7500 / 7500 平衡，step 05 之后仍然显著偏向 safe；unsafe 样本更容易在过滤中被淘汰。

也就是说：
- step 04 的平衡扩充并不能自动保证 step 05 后仍平衡
- unsafe 类样本的 teacher 解释质量仍然更不稳定

---

## 7. 本轮最重要的结论

本轮最重要的结论不是“最终保留了 11626 条”，而是：

> 当前 step 05 的主要损耗来源仍然是 explanation 质量控制问题，而不是单纯数据量不够。

尤其是：
- `evidence_not_grounded = 2160`
- `reason_length_out_of_range = 841`
- `label_mismatch = 373`

这说明当前 teacher 的主要短板依然集中在：
1. evidence 不够稳定地遵守 exact substring 约束
2. reason 经常写得太长
3. 新扩充样本中出现了一定规模的 teacher/gold label 不一致

---

## 8. 丢弃原因分析

### 8.1 `evidence_not_grounded = 2160`

这是本轮最大的丢弃来源，占全部 dropped 的：
- `2160 / 3374 = 64.02%`

这类样本通常不是 teacher 主判断错误，而是：
- evidence 是语义正确的概括
- 但不是原 `question + response` 中的逐字片段
- 或 teacher 做了轻微改写、压缩、重新格式化

这说明：

> teacher 的 reasoning 能力通常强于“精确摘录 evidence span”的能力。

也就是说，当前 teacher 更容易写出“合理解释”，但不总能写出“严格可回指的证据片段”。

这仍然是 step 05 的第一大瓶颈。

### 8.2 `reason_length_out_of_range = 841`

这是第二大丢弃来源，占全部 dropped 的：
- `841 / 3374 = 24.93%`

当前 step 05 的 reason 长度阈值是：
- `min_reason_chars = 20`
- `max_reason_chars = 400`

来源见：
- [scripts/05_filter_pseudo_labels.py:102-103](../scripts/05_filter_pseudo_labels.py#L102-L103)

这批样本大概率主要是：
- reason 太长
- teacher 解释得过满
- 生成风格更像自由分析，而不是稳定、紧凑的 SFT target

这说明：
- 当前 teacher 输出依然不够“收敛”
- 对 step 06 来说，限制 reason 风格仍然是必要的

### 8.3 `label_mismatch = 373`

这是本轮一个值得特别注意的新点。

在旧的 3000 条版本里，step 05 一度已经接近把 label mismatch 压到很低甚至 0；但这次扩充到 15000 条后，又出现了：
- `label_mismatch = 373`

占全部 dropped 的：
- `373 / 3374 = 11.05%`

这说明：
- 新增的 12000 条 teacher 数据里，确实混入了一部分 teacher/gold 不一致样本
- 也就是说，扩大量级后，teacher 的判断稳定性没有完全保持住旧版本水平

这是和旧 3000 条版本相比，最值得额外关注的变化之一。

---

## 9. 这轮结果和旧 3000 条版本相比意味着什么

旧 3000 条版本的核心结论更偏向：
- label mismatch 不是主要问题
- 最大问题是 evidence grounding

而这次 15000 条版本说明情况稍有变化：

### 不变的部分

仍然成立的结论：
- 最大问题依旧是 `evidence_not_grounded`
- reason 过长仍然是第二大问题
- unsafe 比 safe 更容易被过滤掉

### 新变化

本轮新增暴露出的点：
- 扩大量级后，`label_mismatch` 不再是可以忽略的 0 级问题
- 新 teacher 样本里有一批样本在判断方向上与 gold label 不一致

这意味着：
- step 04 的扩充策略在“数量”上成功了
- 但在“新增样本的一致性稳定性”上，仍然存在质量损耗

换句话说：

> 扩充 teacher 数据是成功的，但并不是“无代价扩充”；规模变大后，数据质量分布也发生了变化。

---

## 10. 当前对 step 05 结果的整体判断

### 10.1 主过滤结果是可用的

当前保留下来：
- `11626 / 15000`

这对 MVP 后续的 SFT 已经是相当可观的数据池。

### 10.2 数据量显著提升

和旧 3000 条版本相比，当前最大直接收益是：
- 最终可进入 step 06 的候选监督数据显著增加

这会直接提升：
- SFT 训练集规模上限
- 训练稳定性
- 后续评估可用性

### 10.3 但分布重新失衡了

虽然 step 04 输入是 7500 / 7500，过滤后却变成：
- safe: `6489`
- unsafe: `5137`

因此：
- 后续 step 06 训练时要意识到分布已经偏 safe
- 如果后续很关注 unsafe 表现，可能还需要额外的 rerun 或采样策略补偿

---

## 11. 对后续 rerun 的直接启发

这次 15000 条结果非常适合反向指导下一轮定向 rerun。

### 最值得定向重跑的两类

1. `evidence_not_grounded`
2. `reason_length_out_of_range`

原因：
- 这两类通常不是 teacher 完全判断错了
- 更多是输出风格/格式不满足 strict filter
- 如果重新约束 prompt，理论上有机会提高 keep rate

### 需要单独关注的一类

3. `label_mismatch`

这类不一定适合简单“按同一 prompt 再跑一次”解决，因为：
- 它可能反映 teacher 真实判断偏离 gold label
- 也可能反映部分原始标注与 teacher 判断张力较大

因此，如果后续要做 rerun，建议把：
- `label_mismatch`
- `evidence_not_grounded`
- `reason_length_out_of_range`

分开看，而不是混成一个 dropped 子集统一重跑。

---

## 12. 对 step 06 的直接影响

当前 step 06 建议直接使用：
- [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl)

这是这轮 step 05 的主结果文件。

注意事项：
- 它的总量足够大，可以明显优于旧的 1961 条版本
- 但标签分布已不再平衡
- 如果 step 06 需要更严格控制类别分布，后续应在构造 train/dev 或训练采样时考虑这一点

---

## 13. 当前有效结论

截至本轮，step 05 的最新有效结论如下：

### 工程结论

- step 05 本身逻辑稳定，没有成为本轮主要故障来源
- 当前运行方式建议继续使用显式输出路径，避免覆盖旧结果
- 新主结果文件应优先引用：
  - [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl)

### 数据结论

- 从 15000 条 teacher 数据中保留了 11626 条
- 最大瓶颈仍然是 evidence grounding
- reason 过长仍然是第二大问题
- 扩充到更大规模后，label mismatch 又重新出现，不能忽略
- step 05 之后数据重新偏 safe，unsafe 损耗明显更高

### 流程结论

- step 04 扩容是有价值的，最终有效监督数据量确实显著变大
- 但后续如果继续追求更高 kept 数，重点应放在：
  - 改善 evidence exact span 约束
  - 缩短 reason
  - 降低新增样本里的 label mismatch

---

## 14. 一句话总结

这轮 step 05 的最新结果说明：

> 基于 [pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl) 的正式过滤已经成功产出 [pseudo_filtered_15000.jsonl](../data/processed/pseudo_filtered_15000.jsonl)，最终从 15000 条 teacher 数据中保留了 11626 条；当前最大的质量瓶颈仍然是 evidence 不能严格 grounded，其次是 reason 过长，而扩大量级后重新出现的 373 条 label mismatch 则提示新增 teacher 数据的一致性仍需继续优化。
