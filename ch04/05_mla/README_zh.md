# 多头潜在注意力 (MLA)

**原文链接**: e:\AI\LLMs-from-scratch\ch04\05_mla\README.md
**翻译时间**: 2026-01-27
**文章类型**: 技术文档

---

本节额外材料将为你展示：使用多头潜在注意力 (MLA) 相比传统的多头注意力 (MHA) 能够节省多少内存。

## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论过分组查询注意力 (GQA) —— 它是为了提升 MHA 计算效率而设计的一种巧妙方案。多项消融实验（比如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 中的研究）都表明，在大语言模型的建模性能方面，GQA 的表现与标准 MHA 不相上下。

而现在，被 [DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437) 采用的多头潜在注意力 (MLA)，则提供了一种完全不同的内存节省策略 —— 它与 KV 缓存的配合简直是天作之合。与 GQA 不同（GQA 通过共享键值头来节省内存），MLA 采用了另一种思路：在将键和值张量存入 KV 缓存之前，先把它们"压缩"到一个低维空间中。

等到推理时，这些压缩过的张量会被"解压"回原始尺寸后再使用，如下图所示。这样做虽然增加了一次额外的矩阵乘法运算，但换来的却是内存使用的大幅降低。


![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)


**（顺便提一下：查询向量其实也会被压缩，但那只发生在训练阶段，推理时不需要。）**

说到这里，有必要澄清一点：MLA 并不是 DeepSeek V3 的首创，它的 [前辈 DeepSeek V2](https://arxiv.org/abs/2405.04434) 就已经在使用（甚至正是它引入了）这项技术。另外，V2 论文里还有一些很有意思的消融实验，或许能解释为什么 DeepSeek 团队最终选择了 MLA 而不是 GQA（见下图）。


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" /


从上图可以看出：GQA 的性能似乎比不上 MHA，而 MLA 反而在建模性能上略胜 MHA 一筹 —— 这很可能就是 DeepSeek 团队做出这个选择的原因。（如果能再看到 MLA 和 GQA 在"每个 Token 的 KV 缓存"节省方面的对比数据，那就更有意思了！）

在进入下一个架构组件之前，先用一句话总结一下本节的内容：MLA 是一个巧妙的技术，不仅能减少 KV 缓存的内存占用，在建模性能上甚至还能稍稍超越 MHA。

## MLA 的内存节省

内存的节省主要体现在 KV 存储上。我们可以用下面的公式来计算 KV 存储的大小：

```
bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem
```

相比之下，MHA 的 KV 缓存内存是这样计算的：

```
bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem
```

这意味着什么？在 MLA 中，我们把 "embed_dim × 2 (K,V)" 这个庞大的数字压缩成了 "latent_dim"，因为我们只需要存储压缩后的潜在表示，而不是像上图中那样存储完整的键和值向量。


你可以使用本文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本，在不同的模型配置下测试一下，看看使用 MLA 替代 MHA 到底能省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

注意上面的压缩配置（`--emb_dim 2048 -> latent_dim 1024`），这实现了与 GQA 类似的节省效果。但在实际应用中，压缩率是一个需要仔细调试的超参数 —— 如果把 `latent_dim` 设得太小，可能会对建模性能产生负面影响（这和 GQA 中把 `n_kv_groups` 设得太多是同样的道理）。

下图展示了在不同 `latent_dim` 取值下，MLA 相比 MHA 的内存节省效果（作为上下文长度的函数）：


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" /


你可以通过运行 `uv run plot_memory_estimates_mla.py` 来复现这张图。




## MLA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 这两个脚本，提供了实际的代码示例，让你在 GPT 模型实现的场景下，直观地对比 MHA 和 MLA 的内存使用差异。

这里的 MLA 代码参考了 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 的实现方式。

值得一提的是：MLA 其实可以和 [GQA](../04_gqa) 结合使用，但为了保持代码简洁，这里没有这么做。（目前我也没听说有哪个知名 LLM 这样做。）

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

你可能好奇：为什么这里的内存节省没有像上面图表中那么夸张？原因有两个：

1. 我用的是较小的配置，目的是让模型能在合理时间内完成生成。
2. 更重要的是，这里统计的是整个模型的内存，而不仅仅是注意力机制部分；模型中的全连接层占用了大部分内存（不过那就是另一个分析话题了）。
