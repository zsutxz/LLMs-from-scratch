# 线性注意力的门控 DeltaNet

**原文链接**: e:\AI\LLMs-from-scratch\ch04\08_deltanet\README.md
**翻译时间**: 2026-01-27
**文章类型**: 技术文档

---

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer 架构，实现了对注意力机制的替代方案 —— 其计算复杂度随上下文长度呈线性增长，而非二次方增长。

Qwen3-Next 和 Kimi Linear 都采用了 3:1 的比例，这意味着每三个使用线性门控 DeltaNet 变体的 transformer 块，就搭配一个使用完整注意力的块，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next 对比 Kimi Linear">



## 简介与概览

门控 DeltaNet 是一种线性注意力变体，其灵感来自循环神经网络，包括来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文的门控机制。某种意义上说，门控 DeltaNet 就是带 Mamba 风格门控的 DeltaNet，而 DeltaNet 本身是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention (KDA) 机制修改了 Qwen3-Next 的线性注意力机制，这本质上是对门控 DeltaNet 的改进。Qwen3-Next 使用标量门（每个注意力头一个值）来控制记忆衰减率，而 Kimi Linear 将其替换为针对每个特征维度的通道级门控。根据作者的说法，这提供了更强的记忆控制能力，进而改善了长上下文推理。

此外，对于完整注意力层，Kimi Linear 用多头潜在注意力 (MLA) 替换了 Qwen3-Next 的门控注意力层（本质上是带输出门控的标准多头注意力层）。这就是我们之前在 DeepSeek V3/R1 章节讨论的 MLA 机制，但增加了一个额外的门控。（回顾一下，MLA 通过压缩键/值空间来减少 KV 缓存的大小。）

Kimi Linear 中的 MLA 没有使用门控，这是有意为之，以便作者能够更直接地将架构与标准 MLA 进行比较。不过，他们 [表示](https://x.com/yzhang_cs/status/1984631714464088563) 计划在未来添加这个功能。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，本节额外材料将专注于门控 DeltaNet 方面。


## 门控注意力

在深入门控 DeltaNet 之前，让我们先简单聊聊门控本身。如上图 Qwen3-Next 架构的上半部分所示，Qwen3-Next 使用"门控注意力"。这本质上就是带额外 sigmoid 门控的常规完整注意力。

为了演示说明，我在下面第 3 章的 `MultiHeadAttention` 代码中添加了这个简单的门控修改：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：添加门控
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### 新增：添加门控
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### 新增：添加门控
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



如我们所见，在像往常一样计算注意力之后，模型使用来自同一输入的独立门控信号，应用 sigmoid 将其保持在 0 到 1 之间，然后将其与注意力输出相乘。这允许模型动态地放大或缩小某些特征。Qwen3-Next 开发者 [指出](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)，这有助于训练稳定性：

> [...] 注意力输出门控机制有助于消除注意力下沉 (Attention Sink) 和大规模激活等问题，确保模型在整个过程中的数值稳定性。


## 门控 DeltaNet

那么，什么是门控 DeltaNet？门控 DeltaNet（*Gated Delta Network* 的缩写）是 Qwen3-Next 的线性注意力层，旨在作为标准 softmax 注意力的替代方案。它采用了前面提到的 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文中的设计。

门控 DeltaNet 最初被提出作为 Mamba2 的改进版本，它结合了 Mamba2 的门控衰减机制与 delta 规则。

Mamba 是一种状态空间模型（transformer 的替代方案），这是一个很大的话题，值得在未来单独介绍。

Delta 规则部分指的是计算新值与预测值之间的差异（delta，Δ），以更新用作记忆状态的隐藏状态（稍后会详细介绍）。

（顺便说一句：熟悉经典机器学习文献的读者可以将此理解为类似于受生物学启发的赫布学习："一起激发的细胞连接在一起。"这基本上是感知器更新规则和基于梯度的学习的前身，但没有监督。）

门控 DeltaNet 有一个与前面讨论的门控注意力中的门控类似的门控，只不过它使用 SiLU 而不是 logistic sigmoid 激活，如下图所示。（选择 SiLU 可能是为了改善标准 sigmoid 的梯度流动和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="门控 DeltaNet" width=500px>

然而，如上图所示，门控 DeltaNet 中的"门控"也指的是几个额外的门控：

- `α`（衰减门）控制记忆随时间衰减或重置的速度，
- `β`（更新门）控制新输入修改状态的强度。

在代码中，上面描绘的门控 DeltaNet 的简化版本（没有卷积混合）可以实现如下（代码灵感来自 Qwen3 团队的 [官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)）：

```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：delta 规则和输出门控的门
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # 注意：衰减门 alpha 对应
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # 我们可以这样实现
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # 但为了可解释性和模仿官方实现，我们将偏置分离

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### 新增：计算 delta 规则门控
        beta = torch.sigmoid(self.W_beta(x))
        alpha = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # 新增

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = gate.transpose(1, 2)  # 新增

        ####################################################
        ### 新增：类似 QKNorm 的归一化用于 delta 规则
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### 新增：门控 delta 规则更新
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t.exp()
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
        ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### 新增：应用 RMSNorm 和 SiLU 门控
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（注意：为了简单起见，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合，以保持代码更易读，专注于循环方面。）

所以，如上所示，与标准（或门控）注意力相比，这里有很多不同之处。

在门控注意力中，模型计算所有 token 之间的常规注意力（每个 token 都关注或查看其他所有 token）。然后，在获得注意力输出后，一个门控（sigmoid）决定保留多少输出。要点是，这仍然是随上下文长度呈二次方增长的常规缩放点积注意力。

回顾一下，缩放点积注意力计算为 softmax(QKᵀ)V，其中 Q 和 K 是 *n*×*d* 矩阵，*n* 是输入 token 的数量，*d* 是嵌入维度。所以 QKᵀ 产生一个 *n*×*n* 的注意力矩阵，然后乘以一个 *n*×*d* 维的值矩阵 V：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="二次方注意力" width=500px />

在门控 DeltaNet 中，没有 *n*×*n* 的注意力矩阵。相反，模型逐个处理 token。它维护一个运行记忆（状态），随着每个新 token 的到来而更新。这就是实现为以下代码的部分，其中 `S` 是在每个时间步 *t* 循环更新的状态：

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t.exp()
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

而门控控制着记忆如何变化：

- α（`alpha`）调节遗忘多少旧记忆（衰减）。

- β（`alpha`）调节时间步 *t* 的当前 token 更新记忆的程度。

（而上面代码片段中未显示的最终输出门控类似于门控注意力；它控制保留多少输出。）

所以，从某种意义上说，门控 DeltaNet 中的这种状态更新类似于循环神经网络 (RNN) 的工作方式。优势是它随上下文长度呈线性增长（通过 for 循环），而不是二次方增长。

这种循环状态更新的缺点是，与常规（或门控）注意力相比，它牺牲了来自完整成对注意力的全局上下文建模能力。

门控 DeltaNet 在一定程度上仍然可以捕获上下文，但它必须通过记忆 (*S*) 瓶颈。该记忆大小固定，因此更高效，但它将过去的上下文压缩成类似于 RNN 的单个隐藏状态。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构不会用 DeltaNet 层替换所有注意力层，而是使用前面提到的 3:1 比例。

## DeltaNet 内存节省

在上一节中，我们讨论了 DeltaNet 相比完整注意力的优势，即在计算复杂度方面对上下文长度呈线性而非二次方增长。

除了线性计算复杂度外，DeltaNet 的另一个大优势是内存节省，因为 DeltaNet 模块不会增加 KV 缓存。（有关 KV 缓存的更多信息，请参阅 [../03_kv-cache](../03_kv-cache)）。相反，如前所述，它们维护固定大小的循环状态，因此内存随上下文长度保持恒定。

对于常规多头注意力 (MHA) 层，我们可以这样计算 KV 缓存大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（乘以 2 是因为我们在缓存中存储了键和值两者。）

对于上面实现的简化版 DeltaNet，我们有：

```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

注意 `KV_cache_DeltaNet` 内存大小没有上下文长度 (`n_tokens`) 依赖性。此外，我们只存储记忆状态 S 而不是单独的键和值，因此 `2 × bytes` 变成了 `bytes`。但是请注意，我们现在这里有一个二次方的 `d_head × d_head`。这来自状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常没什么可担心的，因为头维度通常相对较小。例如，在 Qwen3-Next 中它是 128。

带有卷积混合的完整版本稍微复杂一些，包括卷积核大小等，但上面的公式应该说明了门控 DeltaNet 背后的主要趋势和动机。

我们可以通过以下辅助脚本可视化不同上下文长度的内存估算和节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

注意上面的计算将 `head_dim` 设为 `emb_dim / n_heads`。即，2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="门控 DeltaNet 扩展" width=500px>
