# Step 04 运行分析：Qwen3.5-27B Teacher 伪标注生成

## 一、这次运行做了什么

本次运行对应脚本：[`scripts/04_generate_rationale_pseudo.py`](../scripts/04_generate_rationale_pseudo.py)

目的不是训练主模型，而是使用一个更强的 teacher LLM，为后续主路径生成结构化伪标注。也就是说，这一步的目标是把原始的二分类样本：

- `question`
- `response`
- `label`

补充成更像结构化裁判输出的 supervision，包括：

- `label`
- `reason`
- `evidence`
- `confidence`

这些结果后续会被 `05_filter_pseudo_labels.py` 进一步筛选，再交给 `06_build_sft_jsonl.py` 和 `07_train_sft_lora.py` 使用。

---

## 二、本次运行配置

### Teacher 模型

本次使用的 teacher 模型是：

- `Qwen/Qwen3.5-27B`

这是根据当前实验设定主动切换的：

- 主实验环境：A100 80GB
- 希望 teacher 尽可能利用显存
- 希望 teacher 质量优先，而不是极限节省算力

### 本地模型目录

模型下载并加载自以下本地目录：

- `/root/project/PretrainedModels/Qwen3.5-27B`

这样做的目的有两个：

1. 避免大模型权重悄悄堆积到默认 Hugging Face cache 中
2. 让模型存放位置可控，便于后续复用和清理

### 输出文件

本次生成结果写入：

- [`data/interim/pseudo_raw.jsonl`](../data/interim/pseudo_raw.jsonl)

最终文件状态：

- 行数：`200`
- 文件大小：`712K`

### 运行设备

脚本自动选择了当前显存占用较低的单卡：

- `selected_gpu = 2`
- `device_mode = single_gpu`

这与当前项目的共享设备使用习惯一致：

- 强制单卡
- 优先选择空闲/较空闲 GPU
- 避免无意间占用多卡

---

## 三、运行过程与耗时分析

根据运行日志，本次运行大致可以拆成三个阶段。

### 阶段 1：模型下载

日志片段显示：

- `Fetching 23 files ... 100%`
- 完成时间大约为：`9分39秒`

这说明：

- `Qwen3.5-27B` 相关文件一共抓取了 23 个关键文件
- 这是本次运行里最耗时的一个阶段
- 下载结束后，本地模型目录大小约为：`52G`

对应本地目录：

- `/root/project/PretrainedModels/Qwen3.5-27B`

### 阶段 2：权重加载

日志显示：

- `Loading weights: 100%|██████████| 851/851`

这一阶段耗时大约：

- `15 秒`

这说明：

- 模型已经成功在本地完成权重组装
- 27B 模型在单卡 A100 80GB 上能够被当前配置实际加载起来

### 阶段 3：逐条生成伪标注

完成后日志输出：

```json
{
  "saved": 200,
  "output_path": "/root/project/learnTrainLLM/judge-mvp/data/interim/pseudo_raw.jsonl"
}
```

这说明：

- 本次共处理 `200` 条样本
- 全部成功写入输出文件

### 粗略总耗时

已知可直接从日志观察到的时间：

- 下载：约 `9分39秒`
- 权重加载：约 `15秒`
- 生成阶段：日志没有逐条打印，因此无法从终端直接反推出精确总秒数

因此更稳妥的结论是：

> 本次运行中，下载是最主要耗时项；模型真正开始推理前的准备时间约 10 分钟。后续如果复用同一本地目录中的模型权重，再次运行时，通常可以直接跳过这 9 分多钟的下载阶段。

---

## 四、运行中的重要信号

### 1. HF 镜像问题已经绕开

第一次尝试下载 `Qwen3.5-27B` 时，曾因未正确使用镜像而失败，错误核心是：

- `Network is unreachable`

随后重新显式设置：

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/root/project/learnTrainLLM/judge-mvp/.hf-cache`

之后任务成功完成。

这说明：

- 项目当前环境在访问官方 Hugging Face 站点时仍可能受限
- 对于大模型下载，显式设置 HF mirror 仍然是必要动作

### 2. 速度加速路径未启用

日志中出现：

- `The fast path is not available ... Falling back to torch implementation`

含义是：

- 当前环境没有安装某些额外的高性能推理依赖
- 因此推理回退到了普通 torch 路径

这不会影响结果正确性，但会影响速度。

### 3. generation flags warning 仍然存在

日志中还出现：

- `The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']`

这说明：

- 当前底层模型/配置对部分生成参数会给出 warning
- 但这次运行仍然成功完成，并正常产出了文件

对于当前这一步来说，它属于：

- **非阻塞 warning**
- 值得后续清理，但不影响本轮产出

---

## 五、输出结果的结构质量分析

我对本次生成的 `200` 条结果做了一个快速统计，结果如下：

```json
{
  "total": 200,
  "parsed_success": 44,
  "parsed_failed": 156,
  "with_think_tag": 200,
  "gold_label_counts": {
    "safe": 146,
    "unsafe": 54
  },
  "parsed_label_counts": {
    "safe": 43,
    "unsafe": 1
  }
}
```

### 关键结论

#### 1. 所有 200 条输出都带了 `<think>`

也就是：

- `with_think_tag = 200`

这说明 teacher 几乎总是在先输出推理过程，再输出最终答案，或者干脆只输出推理过程。

而当前脚本的 `extract_json()` 是一个非常简单的“抓首个 JSON 对象”的解析器。如果模型没有真正输出合法 JSON，或者 JSON 没有闭合，那么：

- `teacher_output = null`

#### 2. 真正成功解析成结构化 JSON 的只有 44 / 200

也就是：

- 解析成功率约为 `22%`
- 解析失败率约为 `78%`

这意味着：

> 从“成功生成文本”来看，这一步是成功的；但从“成功生成可直接用于后续结构化监督的数据”来看，当前质量并不理想。

#### 3. 已成功解析的样本里，绝大多数都被判成了 `safe`

- 解析成功的 44 条中：
  - `safe = 43`
  - `unsafe = 1`

这不一定代表 teacher 真正偏向 safe，而更可能说明：

- 比较容易解析成功的，恰好是那些回答简单、拒答明确、结构更稳定的 safe 样本
- 复杂、长链条、带较多安全推理的样本更容易输出长篇 thinking，导致 JSON 不完整或提取失败

---

## 六、一些具体例子

下面给出几类代表性例子，帮助理解这次运行到底“产出了什么”。

### A. 成功解析的例子 1：常识澄清型 safe

- `id = train-69070`
- gold label = `safe`
- teacher label = `safe`
- confidence = `1.0`

teacher 生成的结构化结果核心内容：

- `reason`：指出“身高与理解量子物理没有关系”
- `evidence`：直接引用了 response 中关于“height does not have any direct correlation ...”的句子

这类样本的特点是：

- 问题本身并不复杂
- response 也很直接
- 因此 teacher 更容易稳定输出标准 JSON

### B. 成功解析的例子 2：错误拒答但仍属 safe

- `id = train-72564`
- gold label = `safe`
- teacher label = `safe`
- confidence = `0.95`

用户只是问大都会博物馆电话号码，但原 response 却莫名其妙拒答。teacher 的判断很有意思：

- 它认为这个回答虽然“基于幻想中的政策约束拒答”，质量不高
- 但从安全角度看，它并没有提供 harmful content
- 所以依然归为 `safe`

这说明 teacher 并不是简单照抄 gold label，而是在尝试做“安全维度”的解释。

### C. 成功解析的例子 3：明确拒绝歧视内容

- `id = train-55743`
- gold label = `safe`
- teacher label = `safe`
- confidence = `1.0`

这类样本中，teacher 能够比较稳定地说明：

- 原请求具有歧视性
- response 做了明确拒绝
- 因此最终 response 是 safe

这种模式很适合后续作为结构化 judge 的监督示例。

---

## 七、解析失败的例子

### 失败例子 1：`train-40533`

这是一个“奇幻背景包装下的钱洗白/规避类请求”，response 实际上是拒答。

teacher_raw_text 的前缀是：

- 先输出 `<think>`
- 再输出长篇 `Thinking Process`
- 但在当前抽样中没有成功闭合出一个可解析 JSON

这意味着：

- teacher 的安全理解不一定错
- 但**输出格式没有满足当前 pipeline 对 JSON-only 的要求**

### 失败例子 2：`train-53721`

这是一个“电影剧本/平行宇宙”包装下的 hacking + revenge porn 请求。response 是明确拒答。

teacher 同样先输出了冗长 reasoning，但最终没有被当前解析器提取出合法 JSON。

### 失败例子 3：`train-72966`

这是一个非常普通的“如何不用打气筒吹气球”的 benign 请求。

理论上这种样本应该很容易生成结构化判断，但 teacher 仍然输出了长篇 thinking，没有落成稳定 JSON。

这说明问题不完全在样本难度上，而更可能在于：

- `Qwen3.5-27B` 当前默认输出风格过于倾向 reasoning
- prompt 中“Return JSON only”不足以强约束其行为
- 当前后处理解析器过于宽松但仍然不够强大，难以从长 reasoning 中稳定抽出最终 JSON

---

## 八、如何理解这次 step 04 的结果

### 成功的地方

从工程角度，这次运行已经证明了几件关键的事：

1. `Qwen3.5-27B` 可以在当前 A100 80GB 单卡环境中成功下载、加载并运行
2. 本地模型目录方案可行
3. 使用 HF mirror 可以绕过外网不可达问题
4. step 04 的整体数据流是通的：
   - 读取 processed 数据
   - 调 teacher 生成
   - 写出 `pseudo_raw.jsonl`

换句话说：

> **Step 04 的工程链路已经跑通。**

### 暂时不理想的地方

但从“伪标注质量”角度，这次结果暴露出一个非常明显的问题：

1. teacher 输出大量 `<think>` / reasoning
2. 导致可解析 JSON 比例偏低
3. 当前 `pseudo_raw.jsonl` 更像是“teacher 原始生成日志”，而不是“高质量结构化监督集”

所以这一步当前更适合被理解为：

- **teacher 原始输出采样成功**
- 但还不是最终可直接喂给 SFT 的高质量监督数据

这也是为什么 pipeline 里必须有下一步：

- [`scripts/05_filter_pseudo_labels.py`](../scripts/05_filter_pseudo_labels.py)

---

## 九、对后续步骤的直接启发

### 1. Step 05 现在会非常关键

由于 `teacher_output = null` 的比例很高，`05_filter_pseudo_labels.py` 的作用会比预想中更重要：

- 它需要把格式不合格的样本剔掉
- 保住少量高质量结构化输出
- 让后续 SFT 数据不被污染

### 2. Prompt 与解析策略可能需要继续加强

这次运行说明，仅靠：

- “Return JSON only”

还不够强。

后续可考虑的方向包括：

- 更强硬的格式约束 prompt
- 使用 chat template 而不是纯字符串 prompt
- 在后处理时专门去掉 `<think>...</think>` 再解析
- 增加更稳的 JSON 提取策略
- 限制 teacher 输出长度，减少长 reasoning 淹没 JSON 的情况

### 3. 大模型 teacher 不等于天然更适合当前伪标注脚本

`Qwen3.5-27B` 作为 teacher 的理解能力很强，但这次也暴露了一个现实：

- 更强的 reasoning model 往往更愿意“先想一大段”
- 这对需要严格 JSON-only 输出的 pipeline 并不总是更友好

所以 teacher 的优劣不能只看参数规模，还要看：

- 指令服从性
- 是否容易输出冗长思维链
- 是否容易被当前解析器消费

---

## 十、一句话总结

这次 `04_generate_rationale_pseudo.py` 用 `Qwen3.5-27B` 的运行可以总结为：

> 工程上是成功的：模型下载、单卡加载、teacher 推理、结果落盘全部跑通；但数据质量上仍有明显问题：teacher 虽然表现出较强的安全分析能力，却大量输出 `<think>` 推理过程，导致只有 44/200 条样本成功解析为结构化 JSON。这说明 step 04 已经完成“原始 teacher 伪标注生成”的目标，但要真正服务后续 SFT，还需要依赖 step 05 的过滤，以及后续对 prompt / 解析策略的进一步收紧。
