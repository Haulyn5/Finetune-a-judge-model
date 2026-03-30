# Step 04 运行记录：Qwen3.5-27B Teacher 扩充到 15000 条（当前主版本）

## 1. 本文定位

这份笔记以**最新完成且当前有效**的 step 04 结果为主，记录本次 `Qwen/Qwen3.5-27B` teacher 扩充运行的：
- 目标
- 环境与依赖修复
- 运行配置
- 产物位置
- 最终结果
- 对后续 step 05/06 的直接影响

旧的 3000 条主跑与 57 条断点补跑记录仍然有参考价值，但现在已经降级为历史背景；当前后续流程应优先基于 **15000 条版本** 继续推进。

---

## 2. 当前有效的 step 04 主产物

本轮完成后，当前推荐作为 step 04 主输入的文件是：

- [data/interim/pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

这是由：
- 原有 `3000` 条 teacher 数据
- 新增扩充 `12000` 条 teacher 数据

合并得到的 **15000 条** 版本。

相关中间产物：
- 新增生成文件：[data/interim/pseudo_raw_plus_new.jsonl](../data/interim/pseudo_raw_plus_new.jsonl)
- 合并产物原名：[data/interim/pseudo_raw_expanded.jsonl](../data/interim/pseudo_raw_expanded.jsonl)
- 用户最终重命名后的主文件：[data/interim/pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

---

## 3. 本轮任务目标与采样策略

这次扩充不是重新全量生成，而是：

> 在保留已有 teacher 数据的前提下，把 `safe` 与 `unsafe` 都补齐到 `7500` 条；如果随机抽到已经做过 teacher 生成的样本，则直接跳过并复用旧结果。

具体策略：
- 输入训练集：[data/processed/train.jsonl](../data/processed/train.jsonl)
- 旧 teacher 文件：`data/interim/pseudo_raw.jsonl`
- 以 `id` 作为唯一去重键
- 若 `id` 已存在于旧 teacher 文件，则视为已处理，直接跳过
- 只对缺口样本调用 teacher
- safe / unsafe 分别独立随机抽样补足到目标值
- `sampling_seed = 42`

扩充前的 teacher 覆盖量：
- `safe = 1500`
- `unsafe = 1500`

本次新增需求：
- `safe +6000`
- `unsafe +6000`
- 合计新增 `12000`

dry run 结果确认：
- 候选 `safe = 25109`
- 候选 `unsafe = 6031`
- 均可补足到目标
- 已处理跳过 `3000` 条

---

## 4. 本轮使用的脚本与关键实现

### 4.1 使用脚本

原始 step 04 脚本：
- [scripts/04_generate_rationale_pseudo.py](../scripts/04_generate_rationale_pseudo.py)

本轮新增的临时扩充脚本：
- [scripts/04_plus_expand_teacher_targets.py](../scripts/04_plus_expand_teacher_targets.py)

### 4.2 临时扩充脚本的职责

`04_plus_expand_teacher_targets.py` 负责：
- 读取旧 teacher 数据
- 统计当前 safe/unsafe 数量
- 计算 deficit
- 从 `train.jsonl` 中筛选未处理样本
- 先备份旧文件
- 调用 teacher 生成新增样本
- 将新增结果与旧结果按 `id` 合并

### 4.3 本轮脚本关键行为

- 默认目标：`target_safe = 7500`、`target_unsafe = 7500`
- 默认跳过规则：只要 `id` 已存在于旧 teacher 文件，就不重复生成
- 生成结果先写到单独文件，再合并到扩充文件
- 保留 `teacher_raw_text` 与 `teacher_output`，便于 step 05 和后续审计

---

## 5. 本轮环境与依赖修复结论

这次扩充真正花时间排查的，不是数据逻辑，而是运行环境。

### 5.1 第一个问题：旧版 vLLM 无法导入

最初环境里：
- `torch == 2.10.0+cu128`
- `vllm == 0.2.5`

会报：
- `undefined symbol: _ZN3c104cuda9SetDeviceEi`

结论：
- 这是旧版 `vllm` 与当前 `torch/CUDA` 组合不兼容，不是脚本逻辑问题。

### 5.2 第二个问题：锁文件把 vLLM 回滚到旧版本

虽然一度临时升级了 vLLM，但因为项目锁文件仍锁着旧版：
- [uv.lock](../uv.lock) 中原本仍是 `vllm==0.2.5`

所以每次 `uv run` / `uv sync` 后环境又会被拉回旧版。

### 5.3 第三个问题：新 vLLM 与 `transformers>=5` 冲突

项目原本的依赖约束是：
- [pyproject.toml:8-19](../pyproject.toml#L8-L19)

其中曾写成：
- `transformers>=5.2.0`
- `vllm`

但新版本 `vllm` 要求 `transformers<5`，因此锁文件无法正常解析。

### 5.4 最终稳定下来的依赖组合

本轮最终修复为：
- `torch==2.10.0`
- `transformers>=4.56.0,<5`
- `vllm==0.18.0`

当前这组依赖已经能支持本轮 step 04 扩充任务完整跑通。

### 5.5 第四个问题：vLLM 编译/AOT 路径初始化失败

在升级到 `vllm==0.18.0` 后，又遇到一次 engine 初始化失败，错误核心是：
- `FakeTensorMode` 相关 AOT / compile backend 问题

这不是模型文件坏，也不是数据坏，而是 vLLM 的编译路径在当前环境下不稳定。

### 5.6 最终运行策略：关闭 compile，强制 eager

为解决该问题，在扩充脚本里对 vLLM 初始化增加：
- `enforce_eager=True`

效果：
- 禁用 torch.compile / CUDAGraph 相关路径
- 换取更稳但更慢的 teacher 推理

结论：
- **当前项目中，Qwen3.5-27B teacher 扩充任务应优先使用 eager 模式保证稳定性。**

---

## 6. 本轮实际运行配置

最终成功完成的关键配置如下：

- teacher model: `Qwen/Qwen3.5-27B`
- local model dir: `/root/project/learnTrainLLM/PretrainedModels/Qwen3.5-27B`
- input: `data/processed/train.jsonl`
- existing teacher file: `data/interim/pseudo_raw.jsonl`
- generated new rows: `data/interim/pseudo_raw_plus_new.jsonl`
- merged output: `data/interim/pseudo_raw_expanded.jsonl`
- renamed final output: `data/interim/pseudo_raw_15000.jsonl`
- target safe count: `7500`
- target unsafe count: `7500`
- sampling seed: `42`
- batch size: `8`
- max new tokens: `8192`
- max model len: `8192`
- temperature: `0.0`
- gpu memory utilization: `0.9`
- thinking mode: `False`
- vLLM mode: `enforce_eager=True`
- rerun tag: `expand_to_7500_per_label`

提示：
- 虽然 `max_new_tokens=8192` 保留了很宽松的上限，但 teacher 实际输出并没有这么长；这个配置主要是为了避免过早截断。

---

## 7. 本轮备份与安全措施

这次扩充前做了 teacher 文件备份。

可确认的备份文件：
- [backups/teacher_data/pseudo_raw_before_step04_plus_20260329_020458.jsonl](../backups/teacher_data/pseudo_raw_before_step04_plus_20260329_020458.jsonl)
- [backups/teacher_data/pseudo_raw_before_step04_plus_20260329_052842.jsonl](../backups/teacher_data/pseudo_raw_before_step04_plus_20260329_052842.jsonl)

第二个备份对应这次正式完成的主扩充运行。

这意味着：
- 即使后续 step 05 / 06 实验需要回滚，也仍然保留了扩充前的旧 3000 条版本。

---

## 8. 本轮最终结果

任务最终成功完成。

### 8.1 新增样本数

新增生成：
- `safe = 6000`
- `unsafe = 6000`
- 合计 `12000`

### 8.2 合并后总量

扩充前旧 teacher 数据：
- `3000`

新增：
- `12000`

合并后：
- `15000`

### 8.3 最终标签分布

最终达成：
- `safe = 7500`
- `unsafe = 7500`

也就是说，本轮的目标已经被**完整满足**，不存在 shortfall。

### 8.4 合并信息

合并摘要：
- base_count = `3000`
- rerun_count = `12000`
- replaced = `0`
- appended = `12000`
- merged_total = `15000`

说明：
- 本轮没有覆盖旧样本，而是完全追加了新的未处理样本。

### 8.5 文件行数校验

已核对：
- [pseudo_raw_plus_new.jsonl](../data/interim/pseudo_raw_plus_new.jsonl) = `12000` 行
- [pseudo_raw_expanded.jsonl](../data/interim/pseudo_raw_expanded.jsonl) = `15000` 行

用户后续将扩充结果重命名为：
- [pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

因此后续建议统一用 `pseudo_raw_15000.jsonl` 作为新的主产物引用名。

---

## 9. 本轮运行时长与性能结论

本次正式扩充任务的总运行时长：
- `50891.981` 秒
- 约 **14.14 小时**

原因主要有三点：
1. 使用的是 `Qwen3.5-27B`
2. 使用 eager 模式以换取稳定性
3. batch size 仅为 `8`

从工程角度看：
- 这组配置**可跑通**
- 但并不算快
- 如果后续要继续大规模 teacher 扩充，需要把“稳定”和“速度”作为单独优化议题处理，而不能默认当前配置已经是速度最优解

当前结论是：
- **该配置适合保证一次性跑通，不适合作为高吞吐长期生产配置。**

---

## 10. 对 step 05 / step 06 的直接影响

### 10.1 Step 05

现在 step 05 建议优先使用：
- [data/interim/pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

而不是旧的 3000 条版本。

这将直接扩大：
- step 05 的输入规模
- 后续 step 06 SFT 数据池上限

### 10.2 Step 06

如果 step 05 的过滤率与之前同量级，那么：
- 15000 条 teacher 数据应明显提高最终可进入 SFT 的样本量

但仍要注意：
- teacher 数量变大不等于最终高质量样本线性增加
- 真正可用比例仍取决于 step 05 的 grounding、reason length、label consistency 等过滤结果

因此最合理的后续动作是：
1. 基于 `pseudo_raw_15000.jsonl` 运行 step 05
2. 看真实 kept / dropped / drop reasons
3. 再决定是否需要继续 rerun 某些 drop subset 或进一步改 prompt

---

## 11. 与旧 3000 条版本的关系

旧版本并没有失效，只是角色变化了：
- 旧的 `pseudo_raw.jsonl`：现在更像是“首次正式 teacher 运行的历史版本”
- 新的 `pseudo_raw_15000.jsonl`：现在是“后续实验优先使用的当前主版本”

因此后续文档、脚本和实验记录里，如果没有特别说明，建议默认引用：
- [data/interim/pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)

---

## 12. 当前对 step 04 的有效结论

截至本轮，step 04 的最新有效结论如下：

### 工程结论

- 项目执行统一使用 `uv run`
- `Qwen/Qwen3.5-27B` 仍是当前 teacher 主路径
- 对大规模扩充，新增临时脚本 [04_plus_expand_teacher_targets.py](../scripts/04_plus_expand_teacher_targets.py) 是有效工具
- 当前稳定依赖组合是：
  - `torch==2.10.0`
  - `transformers>=4.56.0,<5`
  - `vllm==0.18.0`
- 当前稳定运行模式是：
  - `enforce_eager=True`

### 数据结论

- teacher 数据已经成功从 `3000` 扩充到 `15000`
- 最终标签分布严格平衡：`7500 / 7500`
- 旧样本通过 `id` 复用，没有重复 teacher 调用
- 本轮新增数据与旧数据可按 `id` 安全合并

### 流程结论

- “先 dry run → 再备份 → 再正式生成 → 再合并” 这条流程已验证有效
- 对长任务，应该优先看真实输出日志与真实文件行数，而不是只看任务摘要
- 对大模型 vLLM 推理，稳定性问题通常优先从依赖版本与运行模式入手排查

---

## 13. 后续建议

推荐下一步按下面顺序推进：

1. 基于 [pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl) 正式运行 step 05
2. 统计：
   - kept 总数
   - drop reasons
   - safe/unsafe 保留分布
3. 如果过滤后仍觉得样本量不足，再考虑：
   - 对 step 05 dropped subset 定向 rerun
   - 调整 teacher prompt
   - 或进一步优化生成吞吐

---

## 14. 一句话总结

当前 `judge-mvp` 的 step 04 已经从“3000 条 teacher 主跑 + 断点补跑”升级为一个更完整的扩充版本：

> 使用 `Qwen/Qwen3.5-27B + vLLM 0.18.0 + eager mode`，在跳过已处理样本的前提下，将 teacher 数据成功扩充到 [pseudo_raw_15000.jsonl](../data/interim/pseudo_raw_15000.jsonl)，最终达到 `safe=7500`、`unsafe=7500` 的平衡规模，并保留了完整备份与中间产物，可直接作为后续 step 05 的主输入。
