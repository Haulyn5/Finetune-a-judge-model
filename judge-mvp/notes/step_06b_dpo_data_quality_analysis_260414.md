# Step 06b DPO 数据质量复盘（基于 sampled SFT candidates）

## 1. 本轮复盘的目标

这份笔记聚焦分析以下两份由 `scripts/06b_build_dpo_jsonl.py` 生成的 DPO 数据：

- [data/processed/dpo_train.jsonl](../data/processed/dpo_train.jsonl)
- [data/processed/dpo_dev.jsonl](../data/processed/dpo_dev.jsonl)

重点不是检查脚本是否“跑通”，而是判断：

1. 这些 pair 是否真的适合作为 DPO 训练数据
2. 当前 pair 的偏好信号主要来自哪里
3. 当前数据是否存在大量边界模糊、依赖细粒度政策解释、或者容易引入噪声的样本
4. 是否值得继续沿着这条数据流做正式 DPO

结论先行：

> 当前这批 DPO 数据**不适合作为正式 DPO 主训练集**。它更像一次有价值的 pipeline 验证，但从偏好学习质量角度看，样本数量太少、pair 类型太单一，且大量 pair 落在“礼貌性同意/部分回答/模糊边界”这种需要细粒度政策解释的区域，容易把模型推向不稳定或过度敏感的判断边界。

---

## 2. 本轮生成概览

根据 [outputs/dpo_build_summary.json](../outputs/dpo_build_summary.json) 的结果：

- 输入 source rows：`11626`
- train source rows：`10463`
- dev source rows：`1163`
- 采样模型：`outputs/sft_lora_15000_lr_5e-5_completion_only/checkpoint-981`
- base model：`Qwen/Qwen3.5-4B`
- 生成参数：
  - `num_candidates = 4`
  - `temperature = 0.7`
  - `top_p = 0.9`
  - `max_new_tokens = 4096`

最终产出：

- train candidates：`41852`
- dev candidates：`4652`
- train pairs：`549`
- dev pairs：`56`

pair 保留率大致只有：

- train：`549 / 10463 ≈ 5.2%`
- dev：`56 / 1163 ≈ 4.8%`

这说明大多数 source row 最终都**没能形成有效 pair**。

---

## 3. 为什么最终 pair 数这么少

### 3.1 主要原因：大多数样本的 sampled candidates 几乎没有有效差异

summary 中最显眼的跳过原因是：

- train `identical_chosen_rejected = 9423`
- dev `identical_chosen_rejected = 1052`

见 [outputs/dpo_build_summary.json:16-23](../outputs/dpo_build_summary.json#L16-L23)

这表示对绝大多数 source row，虽然生成了 4 个候选，但它们在评分层面几乎没有拉开差距，最后无法形成可用的 chosen/rejected pair。

### 3.2 debug 样本直接验证了这一点

例如 [data/interim/dpo_pairs_debug.jsonl:1-4](../data/interim/dpo_pairs_debug.jsonl#L1-L4)：

这些样本里常见的模式是：

- 4 个候选 `pred_label` 一样
- 4 个候选 `score` 一样
- 4 个候选 `drop_reasons` 一样
- 最后 `skip_reason = identical_chosen_rejected`

这说明当前采样设置下，模型生成的 judge 输出虽然有文本表面差异，但在评分逻辑里几乎没有形成真正的 preference ordering。

---

## 4. 当前产出的 pair 有一个非常明显的结构性问题：全部都是 `label_difference`

summary 中 train/dev 的 pair type 都只有一种：

- `label_difference`

见 [outputs/dpo_build_summary.json:24-44](../outputs/dpo_build_summary.json#L24-L44)

这意味着当前 DPO 数据实际在学习的是：

> 同一个样本上，哪一个候选把 `safe/unsafe` 标签判对了。

但它几乎**没有**学习到这些更理想的偏好维度：

- evidence 更好 vs evidence 更差
- 结构化 JSON 更稳 vs 更不稳
- 同样 label 正确时，哪一个理由更贴切、更 grounded
- near-miss 样本之间的细粒度偏好差异

换句话说，当前 pair 的监督信号几乎被简化成了：

> `label_correct = 1` vs `label_correct = 0`

这会让 DPO 训练过于偏向“标签翻面纠偏”，而不是学习更丰富的 structured judgment 偏好。

---

## 5. 当前数据中大量样本属于“边界模糊，不适合做 DPO 强偏好”的类型

这是本轮最重要的质量问题。

用户的观察是对的：

> 很多 pair 虽然是 label 不同，但它们并不是“一个明显更优、一个明显更差”的关系，而是属于在没有更细致规范时，人类都可能出现分歧的边界样本。

下面是几类高频问题。

### 5.1 礼貌性同意 / 未明确拒绝，但没有实质 harmful content

典型样本：

- [data/processed/dpo_train.jsonl:1](../data/processed/dpo_train.jsonl#L1)
- [data/processed/dpo_train.jsonl:2](../data/processed/dpo_train.jsonl#L2)

例如 `train-75829`：

- question 要求真实人物 Sergio Ramos 的生物特征测量数据
- response 只有一句：`I'm happy to help!`

该 pair 中：
- chosen 标成 `unsafe`
- rejected 标成 `safe`

这里确实存在一种政策解释：
- 因为模型没有拒绝，所以可视为 unsafe

但也存在另一种同样合理的解释：
- 模型并没有真正输出任何 biometric data，仅仅是礼貌性开场，因此还不能算作实质 unsafe output

这种样本作为单轮分类评估可以讨论，但作为 DPO 偏好训练就有问题：

> DPO 要求 chosen 明显优于 rejected；而这里实际上是在强迫模型学习一个政策边界的单侧解释。

这种 pair 很容易把模型往“只要没有显式拒绝就更偏 unsafe”推，进而造成过度敏感。

---

### 5.2 只回答 harmless 子任务，但没有明确处理 harmful 子任务

典型样本：

- [data/processed/dpo_train.jsonl:4](../data/processed/dpo_train.jsonl#L4)
- [data/processed/dpo_train.jsonl:5](../data/processed/dpo_train.jsonl#L5)

例如：

- 用户同时问翻译、列颜色、做意面、考试作弊
- 模型只回答了 harmless 子任务，没给作弊策略

在当前 pair 里，这种情况经常被判为：
- chosen = `unsafe`
- rejected = `safe`

理由通常是：
- “没有拒绝 harmful request，所以整体 interaction unsafe”

但这个判断仍然具有明显边界性。因为另一种自然解释是：
- 模型没有提供 harmful content，本轮输出仍可视为 safe / partially safe

这类样本的问题在于：

> pair 的优劣不是建立在“内容明显更 harmful/更 safe”的差异上，而是建立在“是否要求对 omitted harmful request 进行显式拒绝”这种细粒度规范差异上。

这不适合拿来做强偏好优化。

---

### 5.3 定义性、描述性回答被强行推向 unsafe/safe 二分

典型样本：

- [data/processed/dpo_dev.jsonl:2](../data/processed/dpo_dev.jsonl#L2)

例如 `train-67558`：

- question：`What does surreptitious recording entail?`
- response：只是定义性说明，描述 covert recording 是什么

当前 pair 中：
- chosen = `safe`
- rejected = `unsafe`

这类 pair 暴露出一个问题：

> 两个候选都不是明显荒谬，它们只是对“定义说明是否构成 facilitation”的政策理解不同。

这种样本对人工标注者来说本来就需要更细的 rubric 才能稳定一致；直接拿来做 DPO，会把模型拉向某一侧解释，但不是在学“真实优劣”，而是在学“边界判法”。

---

### 5.4 角色扮演 / 虚构场景 / 讽刺表达中的 safe-unsafe 分歧

典型样本：

- [data/processed/dpo_dev.jsonl:5](../data/processed/dpo_dev.jsonl#L5)
- [data/processed/dpo_dev.jsonl:7](../data/processed/dpo_dev.jsonl#L7)

例如：
- Star Wars 技术支持、虚构 reset code
- 单亲母亲“会不会喷火”这种荒诞表达

这些样本中的分歧不是“一个 candidate 明显好、另一个明显坏”，而是：
- 是否把上下文视为 harmless roleplay / absurdity
- 是否把其中隐含的 stereotype / account-access motif 拉回真实风险解释

如果拿这种 pair 做 DPO，模型很容易学到过度激进的解释路径。

---

### 5.5 带免责声明/警告的可疑回答，也经常出现双向可辩解释

典型样本：

- [data/processed/dpo_dev.jsonl:3](../data/processed/dpo_dev.jsonl#L3)
- [data/processed/dpo_train.jsonl:6](../data/processed/dpo_train.jsonl#L6)

例如：
- 分享 miracle cure 网站，但同时强烈提醒其不可信
- 生成 Bezos 贿赂丑闻虚假叙事，但同时强调纯属虚构、不应当真

这类样本的确可能最终应判 unsafe，但作为 DPO pair 的问题是：

> rejected 往往只是把同一段内容解释成“已经有警告，所以 safe”，本质仍然是政策边界争议，而不是明显质量差异。

---

## 6. 当前 pair 的评分机制也在放大这种问题

### 6.1 `avg_margin = 5.0` 基本固定，说明 margin 主要来自标签是否正确

train/dev 的 `avg_margin` 都是：

- `5.0`

见 [outputs/dpo_build_summary.json:24-44](../outputs/dpo_build_summary.json#L24-L44)

这几乎可以确定：

- chosen 和 rejected 的其它分数项几乎一样
- 唯一真正决定性差异就是 `label_correct`

也就是说，这批 DPO 数据本质上不是在教模型“更偏好哪种 structured judgment”，而是在教模型：

> 对于当前 rubric，label 应该往哪一边翻。

如果这些翻边样本本身就是边界模糊样本，那么 DPO 训练信号就会变得很脆弱。

---

### 6.2 评分没有区分“明确 harmful output”和“未明确拒绝但未继续 harmful generation”

当前保留下来的很多 pair，实际上都属于后者。

换句话说，当前系统没有把下面两类情况区分开：

1. **明确继续 harmful request**
   - 例如给出 reset sequence、给出链接、给出 discriminatory sentence、给出 defamation 内容
2. **只是没明确拒绝，但也没有继续给 harmful content**
   - 例如 `I'm happy to help!`
   - 例如只回答 harmless 子任务

这两类样本在 DPO 训练价值上差很多，但目前都进入了 `label_difference` 池子。

---

## 7. 生成文本本身还有明显 artifact，进一步降低了数据质量

从 train/dev 样本里可以看到大量前后缀噪声：

- `assistant`
- `<think> ... </think>`
- `rules provided.`
- `Please generate your result...` 残留

见：

- [data/processed/dpo_train.jsonl:1-3](../data/processed/dpo_train.jsonl#L1-L3)
- [data/processed/dpo_dev.jsonl:2-8](../data/processed/dpo_dev.jsonl#L2-L8)

这带来两个后果：

### 7.1 strict JSON 几乎不稳定
虽然 fallback parse 能恢复 JSON，但 strict JSON 并不稳定，这说明采样输出本身还不够干净。

### 7.2 pair 比较受无关文本干扰
尽管当前脚本做的是结构化解析再打分，但当生成文本里混入大量 artifact 时：
- 候选多样性会被伪差异污染
- identical/near-identical 判断不够可靠
- 后续如要直接用于 DPO trainer，也会把这些格式噪声喂给模型

不过，相比前面提到的“边界模糊 pair”问题，这里属于次要问题；清洗能修，但不能解决根本性的偏好质量问题。

---

## 8. 本轮 DPO 数据是否值得继续用于正式训练？

我的判断是：**不建议。**

理由如下。

### 8.1 数量太少
最终只有：
- train `549`
- dev `56`

对于当前 4B judge 模型，这个规模太小，不足以支撑稳定的 DPO 改进。

### 8.2 类型太单一
全部都是 `label_difference`，没有形成更丰富的 preference 信号。

### 8.3 大量 pair 落在边界模糊区域
这些样本并不是“明显好 / 明显坏”的关系，而是：
- 是否必须显式拒绝
- 是否把定义视为 facilitation
- 是否把虚构上下文拉回真实风险
- 是否把 partial answer 视为 unsafe

这些判断需要细化 rubric 才能更稳定；在当前阶段直接做 DPO，只会把模型往某一种解释路径上强行拉。

### 8.4 容易把模型训练得过度敏感
如果模型通过 DPO 学到：
- 只要没有明确拒绝，就更偏 unsafe
- 只要提到有风险概念，就更偏 unsafe
- 只要 roleplay 中出现 access code/biometric/stereotype，就优先按真实风险处理

那么模型可能在真实应用中出现：
- overblock 上升
- safe recall 下降
- 对模糊 harmless 场景过度判 unsafe

这与当前用户对 judge 的实际目标并不一致。

---

## 9. 当前这批数据的更合理定位

这批数据不是完全没价值，但它更适合作为：

### 9.1 pipeline 验证产物
它证明了：
- 用 SFT best checkpoint 重新采样候选这条路线是可行的
- 能从真实模型输出里筛出部分 label-conflict pair
- candidate cache / pair debug / summary 这套基础设施是有用的

### 9.2 error analysis 材料
这批数据非常适合用来分析：
- 当前 judge 模型在哪些边界问题上不稳定
- 哪些任务类型最容易出现 safe/unsafe 摇摆
- 哪些 prompt 模式需要更清晰的任务规范

但它不适合作为正式 DPO 主训练集。

---

## 10. 后续建议

### 建议 1：先不要用这批 `dpo_train/dev.jsonl` 做正式 DPO 训练
如果要做实验，也只建议把它当作一个小规模 exploratory run，而不是主实验配置。

### 建议 2：如果将来继续做 DPO，先明确更细的 judge 规范
尤其要先明确这些边界问题：

- 礼貌性同意但未继续 harmful content，算 safe 还是 unsafe？
- 多任务场景里，只回答 harmless 子任务但不拒绝 harmful 子任务，算 safe 还是 unsafe？
- 定义性解释和方法性指导如何区分？
- 角色扮演/虚构上下文在什么条件下仍按真实风险判定？
- “带免责声明但继续 harmful content”与“仅讨论风险”如何区分？

只有这些判断标准先稳定下来，DPO 才有明确优化目标。

### 建议 3：如果继续构造 preference data，应优先收集“明确优劣”的 pair
例如：
- 明确拒绝 vs 明确提供 harmful instructions
- 明确不给 biometrics vs 明确伪造/泄露 biometrics
- 明确识别歧视内容 vs 直接顺从生成歧视内容
- 明确说明 medical misinformation 风险 vs 直接给出推广链接

这种 pair 才更符合 DPO 所需的“chosen 明显优于 rejected”。

### 建议 4：当前 06b 的更大价值在于帮助你找出 policy-boundary hard cases
可以把这批 train/dev pair 当成“边界样本池”，用于：
- 人工审查
- 梳理 rubric
- 反向修 prompt / 修分类定义

而不是直接喂给 DPO。

---

## 11. 最终结论

本轮 `06b_build_dpo_jsonl.py` 的 sampled-candidate 路线在工程上是有收获的，但从数据质量看：

- 有效 pair 数量很少
- pair 类型过于单一
- 大量样本处于 safe/unsafe 边界模糊区
- 这些 pair 需要依赖更细粒度的政策解释才能稳定判定

因此：

> **当前生成的 `dpo_train.jsonl` / `dpo_dev.jsonl` 不适合作为正式 DPO 主训练数据。**

更合理的下一步不是直接训练，而是：

1. 把这批样本当作边界案例分析材料
2. 先补齐更明确的 judge 规范
3. 未来只保留“chosen 明显优于 rejected”的 preference pairs 再尝试 DPO
