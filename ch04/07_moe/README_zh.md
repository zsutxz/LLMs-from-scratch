# 混合专家模型 (MoE)

**原文链接**: e:\AI\LLMs-from-scratch\ch04\07_moe\README.md
**翻译时间**: 2026-01-27
**文章类型**: 技术文档

---

本节额外材料将为你展示：使用混合专家层 (MoE) 替代常规前馈层 (FFN) 时，每个 token 的内存节省情况。



## 简介

MoE 的核心思想是：用多个专家层替换 transformer 块中的每个前馈模块，而这些专家层本身也都是前馈模块。换句话说，我们把一个前馈块变成了多个前馈块，如下图所示：



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="MoE" width="800px" />

transformer 块内部的前馈块（上图中用深灰色块表示）通常包含了模型总参数中的大部分。（注意：transformer 块，进而前馈块，在 LLM 中会重复很多次；以 DeepSeek-V3 为例，重复了 61 次。）

那么，用 **多个** 前馈块替换 **单个** 前馈块（正如 MoE 设置中所做的），会大幅增加模型的总参数数量。但是，关键技巧在于：我们不会对每个 token 都使用（"激活"）所有专家。相反，一个路由器会为每个 token 只选择一小部分专家。

因为同一时间只有少数专家处于活跃状态，MoE 模块通常被称为 **稀疏** 模块，这与始终使用全部参数集的 **稠密** 模块形成对比。然而，MoE 带来的庞大参数总数提升了 LLM 的容量，这意味着它可以在训练时吸收更多知识。不过，由于我们不会同时使用所有参数，稀疏性让推理过程依然保持高效。

举个例子，DeepSeek-V3 每个 MoE 模块有 256 个专家，总参数量达到 6710 亿。但在推理过程中，同一时间只有 9 个专家是活跃的（1 个共享专家 + 8 个由路由器选中的专家）。这意味着每个 token 的推理步骤只用到 370 亿参数，而不是全部的 6710 亿。

DeepSeek-V3 的 MoE 设计中有一个显著特点：使用共享专家。这是一个对所有 token 都始终活跃的专家。这个想法并不新鲜，早在 [2022 年的 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 年的 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就已经引入了。


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE 共享专家" width="500px" />

（图自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文，已添加注释。）


共享专家的好处最早在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中被注意到，他们发现相比没有共享专家的情况，共享专家能提升整体建模性能。这很可能是因为常见或重复的模式不需要由多个独立专家分别学习，从而让它们有更多空间去学习更专业的模式。


## 混合专家模型 (MoE) 的内存节省

MoE 模型的内存节省主要来自减少的激活存储和计算量。在常规（稠密）前馈层 (FFN) 中，每个 token 都会激活完整的中间维度。

相比之下，MoE 层为每个 token 只通过一小部分专家（例如，从 `num_experts` 个专家中选 `top_k` 个）进行路由。

当使用 MoE 层时，每个 token 只有 `top_k` 个专家是活跃的，所以与具有相同总容量的稠密 FFN 相比，有效内存（和计算量）大约按 `top_k / num_experts` 的比例缩放。


你可以使用本文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本，在不同的模型配置下测试一下，看看使用 MoE 替代 FFN 到底能省多少内存（注意：这是针对单个 transformer 块的，要获得总节省量，需要乘以你模型中 transformer 块的数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

所以，根据上面的结果，我们可以看到：如果我们有一个输入/输出维度 (`emb_dim`) 为 7,168、中间大小 (`hidden_dim`) 为 14,336 的 FFN，这一层大约有 3.08 亿参数，而在前向传播中所有这些参数都是活跃的。

现在，如果我们使用一个总参数数量大致相同（约 3.08 亿）的 MoE 层，配置为 8 个专家、每次激活 2 个，那么每次前向传播只有约 7700 万参数是活跃的。

此外，在专家数量恒定的情况下，我们拥有的专家越多，活跃参数的数量就越少，"节省"的效果就越明显：




<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="MoE 内存节省" width="500px" />




你可以通过以下命令复现这张图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


## MoE 代码示例

本文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 这两个脚本，提供了实际的代码示例，让你在 GPT 模型实现的场景下，直观地对比常规 FFN 和 MoE 的内存使用差异。注意，这两个脚本都使用了 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块，如本页第一张图所示（传统的 GPT-2 使用的是 GELU）。

**注意：这个模型还没有经过训练，所以生成的文本是毫无意义的。你可以在额外材料中找到训练好的 MoE 模型：[../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb)。**



首先，让我们运行一个使用常规 FFN 的模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平对比，我们需要缩小专家的尺寸。例如，如果我们使用 32 个专家，就需要设置 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

我们可以看到：稠密前馈层处理一个 token 大约需要 0.76 毫秒，使用约 0.19 MB 的激活值（峰值接近 0.75 MB）。

稀疏 MoE 层只占用约 0.04 MB 的内存（峰值为 0.11）。但这付出的代价是计算时间大约翻倍。（这是因为增加了路由开销，而且我的实现可能也不是最高效的。）

无论哪种情况，整体生成的 GPU 内存峰值都在 11.5 GB 左右，因为两个版本加载的权重参数数量相同，而且 KV 缓存大小也相同 —— 这两项在这里占了主导地位。

不管怎样，我们都能看到这里的权衡：MoE 将 FFN 内存减少了约 4-5 倍，但前馈计算时间大约增加了一倍。

注意：如果我们一次处理更多 token，比如使用大于 1 的批大小（这里为了代码简洁没有使用批处理），节省效果会更加明显。



