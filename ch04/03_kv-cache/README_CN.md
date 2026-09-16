# 额外材料：KV 缓存



**本文件夹实现了向 GPT 模型添加 KV 缓存的功能。**


## 概述

简而言之，KV 缓存用于存储中间的键（K）和值（V）计算结果以便在推理过程中重用，这可以在生成响应时带来显著的速度提升。缺点是它增加了代码的复杂性，增加了内存使用量，并且不能在训练期间使用。然而，在部署 LLM 时，推理速度的提升往往值得在代码复杂性和内存方面做出权衡。


## 工作原理

想象一下 LLM 正在生成一些文本。具体来说，假设 LLM 被给出以下提示："Time flies"（时光飞逝）。

下图显示了底层注意力分数计算的一个摘录，使用了来自第 3 章的修改图形，其中键和值向量被高亮显示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中学到的，LLM 一次生成一个单词（或 token）。假设 LLM 生成了单词 "fast"（快），使得下一轮的提示变成 "Time flies fast"。这在下图中得到了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们所见，通过比较前两张图，前两个 token 的键和值向量是完全相同的，在每一轮下一个 token 的文本生成中重新计算它们将是浪费的。

因此，KV 缓存的思想是实现一种缓存机制，存储先前生成的键和值向量以供重用，这有助于我们避免不必要的重新计算。


## KV 缓存实现

实现 KV 缓存有很多方法，主要思想是我们在每个生成步骤中只为新生成的 token 计算键和值张量。

我选择了一种简单的方法，强调代码的可读性。我认为直接浏览代码更改来看它是如何实现的是最简单的。

本文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：从第 3 章和第 4 章提取的独立代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但进行了必要的更改以实现 KV 缓存。

您可以：

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件并查找标记新更改的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 通过您选择的文件差异工具查看这两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，这里有一个简短的演练。


### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数内部，我们添加两个缓冲区 `cache_k` 和 `cache_v`，它们将保存跨步骤的连接键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```


### 2. 使用 `use_cache` 标志的前向传递

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法以接受 `use_cache` 参数。在将新的 token 块投影到 `keys_new`、`values_new` 和 `queries` 后，我们要么初始化 kv 缓存，要么追加到我们的缓存中：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # 形状：(b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new

    # ...

    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```



### 3. 清除缓存

在生成文本时，在独立序列之间（例如两次文本生成调用之间），我们必须重置两个缓冲区，因此我们还向 `MultiHeadAttention` 类添加了缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```


### 4. 在完整模型中传播 `use_cache`

在进行了 `MultiHeadAttention` 类的更改后，我们现在修改 `GPTModel` 类。首先，我们向构造函数添加 token 索引的位置跟踪：

```python
self.current_pos = 0
```

然后，我们将单行块调用替换为显式循环，将 `use_cache` 传递到每个 transformer 块：

```python
def forward(self, in_idx, use_cache=False):
    # ...

    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )

    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行小的修改以接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们向 `GPTModel` 添加模型级别的重置，以便一次性清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```


### 5. 在生成中使用缓存

在对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 进行更改后，最后，这是我们在简单的文本生成函数中使用 KV 缓存的方式：

```python
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 使用完整提示初始化缓存
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) 选择具有最高对数概率的 token（贪婪采样）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 将其追加到运行序列中
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 仅向模型提供新 token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，在 c) 中我们只向模型提供新 token，通过 `logits = model(next_idx, use_cache=True)`。在没有缓存的情况下，我们向模型提供整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)`，因为它没有存储的键和值可重用。


## 简单的性能比较

在概念层面介绍了 KV 缓存后，最大的问题是在小示例中它在实践中实际上表现如何。为了尝试这个实现，我们可以将前面提到的两个代码文件作为 Python 脚本运行，这将运行小型 124M 参数 LLM 生成 200 个新 token（给定一个 4 token 提示 "Hello, I am" 开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备 M4 芯片（CPU）的 Mac Mini 上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，正如我们所见，对于小型 124M 参数模型和短 200 token 序列长度，我们已经获得了约 5 倍的加速。（请注意，此实现针对代码可读性进行了优化，而不是针对 CUDA 或 MPS 运行时速度进行优化，后者需要预分配张量而不是重新创建和连接它们。）

**注意：** 模型在两种情况下都生成"乱码"，即看起来像这样的文本：

> 输出文本：Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章将训练模型，您可以在训练后的模型上使用 KV 缓存（但是，KV 缓存仅用于推理期间）来生成连贯的文本。在这里，我们使用未训练的模型以保持代码简单（更简单）。

但更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 实现产生完全相同的文本。这告诉我们 KV 缓存实现正确——很容易犯索引错误，导致结果分歧。



## KV 缓存的优缺点

随着序列长度的增加，KV 缓存的好处和缺点变得更加明显，具体如下：

- [好] **计算效率提高**：没有缓存，步骤 *t* 处的注意力必须将新查询与 *t* 个先前的键进行比较，因此累积工作按二次方扩展，O(n²)。使用缓存，每个键和值只计算一次然后重用，将每步总复杂性降低为线性，O(n)。

- [坏] **内存使用线性增加**：每个新 token 都追加到 KV 缓存中。对于长序列和较大的 LLM，累积的 KV 缓存会变大，这可能会消耗大量甚至过多的（GPU）内存。作为一种解决方法，我们可以截断 KV 缓存，但这会增加更多的复杂性（但同样，在部署 LLM 时，它可能非常值得。）



## 优化 KV 缓存实现

虽然我在上面对 KV 缓存的概念实现有助于清晰度，主要针对代码可读性和教育目的，但在现实场景中部署它（特别是对于较大的模型和更长的序列长度）需要更仔细的优化。


### 扩展缓存时的常见陷阱

- **内存碎片和重复分配**：如前所示通过 `torch.cat` 连续连接张量，由于频繁的内存分配和重新分配会导致性能瓶颈。

- **内存使用线性增长**：没有适当的处理，KV 缓存大小对于非常长的序列变得不切实际。

#### 提示 1：预分配内存

我们可以根据预期的最大序列长度预分配一个足够大的张量，而不是重复连接张量。这确保了一致的内存使用并减少了开销。在伪代码中，这可能如下所示：

```python
# 键和值的预分配示例
max_seq_len = 1024  # 最大预期序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理期间，我们可以简单地将数据写入这些预分配张量的切片中。


#### 提示 2：通过滑动窗口截断缓存

为了避免 GPU 内存爆炸，我们可以实现带有动态截断的滑动窗口方法。通过滑动窗口，我们在缓存中只保留最后 `window_size` 个 token：


```python
# 滑动窗口缓存实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```


#### 实践中的优化

您可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在配备 M4 芯片（CPU）的 Mac Mini 上，生成 200 个 token 且窗口大小等于上下文长度（以保证相同结果）的情况下，代码运行时比较如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，在 CUDA 设备上，速度优势消失了，因为这是一个 tiny 模型，设备传输和通信超过了这个小模型的 KV 缓存的好处。


## 其他资源

1. [Qwen3 从零开始的 KV 缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 从零开始的 KV 缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从零开始理解和编码 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 此 README 的更详细说明
