# 分组查询注意力（GQA）

本额外材料说明了使用分组查询注意力（GQA）相比常规多头注意力（MHA）时的内存节省。


## 简介

近年来，分组查询注意力（GQA）已成为新的标准替代方案，作为比多头注意力（MHA）更高效计算和参数效率的替代方案。请注意，它并不新鲜，可以追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。甚至旧版 Llama 2 系列中的较大变体也使用了它。

这是 GQA 的简要总结。与 MHA 不同（MHA 中每个头都有自己的键和值集合），为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影。

例如，如下图进一步说明的那样，如果有 3 个键值组和 6 个注意力头，那么头 1 和 2 共享一组键和值，而头 3 和 4 以及头 5 和 6 分别共享另一组。


![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)


这种键和值的共享减少了键和值计算的总数，从而降低了内存使用并提高了效率。

因此，总结来说，GQA 背后的核心思想是通过跨多个查询头共享键和值头来减少它们的数量。这（1）降低了模型的参数数量，（2）减少了推理期间键和值张量的内存带宽使用，因为需要从 KV 缓存中存储和检索的键和值更少。

虽然 GQA 主要是 MHA 的计算效率变通方法，但消融研究（如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的那些）表明，在 LLM 建模性能方面，它的表现与标准 MHA 相当。

然而，这假设键值组的数量是仔细选择的。在所有注意力头共享单个键值组的极端情况下（称为多查询注意力），内存使用量甚至更急剧地减少，但建模性能可能会受到影响。（而且，在另一个极端，如果我们设置键值组的数量等于查询头的数量，我们就回到了标准多头注意力。）


## GQA 内存节省

内存节省主要体现在 KV 存储中。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

您可以使用本文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本将其应用于不同的模型配置，以查看通过使用 GQA 而不是 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

使用 GQA 相比 MHA 的节省在下图中进一步显示，针对不同键值组大小作为上下文长度的函数：


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />


您可以通过 `uv run plot_memory_estimates_gqa.py` 重新生成该图。


## GQA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了在 GPT 模型实现背景下比较 MHA 和 GQA 内存使用的实际示例。

请注意，GQA 也用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 额外材料中。然而，为了简单起见，本文件夹中的代码脚本修改了传统上不使用 GQA 的 GPT 架构。

请注意，模型未经训练，因此生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的直接替代品并对其进行训练。

此外，此实现使用了 [另一个额外部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有看到像上面图表中那样大的节省的原因是双重的：

1. 我使用了较小的配置，以便模型在合理的时间内完成生成。
2. 更重要的是，我们在这里查看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个单独分析的主题）。
