# 滑动窗口注意力 (SWA)

**原文链接**: e:\AI\LLMs-from-scratch\ch04\06_swa\README.md
**翻译时间**: 2026-01-27
**文章类型**: 技术文档

---

本节额外材料将为你展示：使用滑动窗口注意力 (SWA) 相比传统的多头注意力 (MHA) 能够节省多少内存。



## 简介

什么是滑动窗口注意力 (SWA)？如果把常规的自注意力看作是一种 **全局** 注意力机制 —— 毕竟每个序列元素都能关注到其他所有元素 —— 那么 SWA 就可以理解为 **局部** 注意力，因为我们把每个查询位置周围的上下文大小限制在一定范围内。下图直观地展示了这种差异。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="滑动窗口注意力" width="500px" />

如上图所示，每个 token 不再关注之前的所有 token，而是只关注自己位置周围的一个固定大小的局部窗口。这种局部化的注意力机制能够大幅降低 KV 缓存的体积。

在本节的剩余部分，我们将以 [Gemma 3](https://arxiv.org/abs/2503.19786) 为例来讨论 SWA —— 这个模型在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中有从零开始的实现。

滑动窗口注意力最早是在 2020 年的 [LongFormer 论文](https://arxiv.org/abs/2004.05150)中提出的，但我们之所以聚焦于 Google 的 Gemma 系列模型，是因为它们是优秀的开源权重模型，充分证明了滑动窗口注意力在近期的先进模型中确实是一个可行且实用的方案。

[Gemma 2](https://arxiv.org/abs/2408.00118) 采用了一种混合策略：局部（滑动窗口）注意力层和全局注意力层按 1:1 的比例交替出现。每个 token 可以关注到 4096 个 token 的上下文窗口。这种 1:1 混合设计的目的，是在效率和全局上下文建模之间找到平衡点 —— 毕竟只用局部注意力的 LLM 可能会显得太过局限。

到了 [Gemma 3](https://arxiv.org/abs/2503.19786)，Google 把这个设计进一步向效率方向推进。它将滑动窗口层和完整注意力层的比例调整到了 5:1，也就是说，每 5 个局部注意力层才配 1 个全局层。此外，滑动窗口的大小也从 Gemma 2 的 4096 个 token 缩小到了 Gemma 3 的 1024 个 token。

有意思的是，Gemma 3 技术报告中的消融实验显示：这些改动对整体模型质量的影响微乎其微。换句话说，通过滑动窗口注意力实现的可观内存和计算节省，在建模性能上的代价却非常小。



## 滑动窗口注意力 (SWA) 的内存节省

内存的节省主要体现在 KV 存储上。我们可以用下面的公式来计算 KV 存储的大小：

```
bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads
```

当使用 SWA 时，我们将上面的序列长度 (seqlen) 替换为窗口大小 W。也就是说，使用滑动窗口注意力时，KV 缓存的大小会按照 "W / seqlen" 的比例缩小。（注意：为了简化，这里假设每一层都使用了滑动窗口注意力。）


你可以使用本文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，在不同的模型配置下测试一下，看看使用 SWA 替代 MHA 到底能省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
MHA + GQA (Ratio: 5:1) : 0.78 GB
```

注意：Gemma 3 是同时使用 SWA 和 GQA 的。

下图展示了在不同上下文长度下，使用 SWA 相比 MHA 的内存节省效果：


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" /


你可以通过以下命令复现这张图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


## SWA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 这两个脚本，提供了实际的代码示例，让你在 GPT 模型实现的场景下，直观地对比 MHA 和 SWA 的内存使用差异。

值得一提的是：SWA 其实可以和 MLA、GQA 结合使用（正如前面提到的），但为了保持代码简洁，这里没有这么做。

另外需要提醒：这个模型还没有经过训练，所以生成的文本是毫无意义的。不过你可以把它当作标准 GPT 模型的直接替代品，在第 5-7 章中用它来进行训练。

最后一点：这个实现使用了 [另一个额外章节](../03_kv-cache) 中介绍的 KV 缓存，所以内存节省的效果会更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # 像 Gemma 3 那样

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

你可能好奇：为什么这里的内存节省没有像上面图表中那么夸张？原因有两个：

1. 我用的是较小的配置，目的是让模型能在合理时间内完成生成。
2. 更重要的是，这里统计的是整个模型的内存，而不仅仅是注意力机制部分；模型中的全连接层占用了大部分内存（不过那就是另一个分析话题了）。
