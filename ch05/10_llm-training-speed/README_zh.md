# PyTorch 性能优化技巧：让 LLM 训练更快速



请注意，本书的编写目的是为了教育，因此原始代码特意保持简单。这样做是为了提高可读性，并确保在不同硬件（包括 CPU 和 GPU）上的兼容性。不过，你可能对一些更高级的 PyTorch 和 GPU 功能感兴趣，以便让 LLM 训练变得更加高效。

本文件夹包含三个代码文件，展示了针对第 5 章中介绍的 LLM 和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：第 5 章的原始代码，用于 CPU 和单 GPU 训练。
   ➤ 运行方式：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：针对单 GPU 训练的优化版本。
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多 GPU 训练优化版本。
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`
   （**注意：** 为了保持与 `01_opt_single_gpu.py` 相比改动最小，此脚本仅支持通过 `torchrun` 进行多进程处理。这意味着不支持通过 `python 02_opt_multi_gpu_ddp.py` 使用多 GPU）

**请注意，这些修改将训练速度从每秒 12,525 个 token（单张 A100）提升到每秒 142,156 个 token（单张 A100）和每秒 419,259 个 token（4 张 A100）。**

我计划在未来更详细地阐述这些差异。目前，查看代码添加了哪些改进的最简单方法是在 Visual Studio Code 中打开这些文件，并通过"比较选定的文件"功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## 单 GPU 速度对比

如上所述，我计划在未来详细说明这些更改。目前，本节包含每个修改的简单性能概述，以 token/秒为单位。所有实验均在 A100 GPU 上运行。

&nbsp;
### 基准版本

请注意，`00_orig.py` 作为基准版本，除了以下几点外，不包含任何重大修改，使用了第 5 章的代码：

- 上下文长度扩大 4 倍（这解释了 `00_orig.py` 与第 5 章相比相对较大的内存占用）；
- 批次大小变化 4 倍（这是 `00_orig.py` 内存占用相对较大的另一个原因）；
- 使用更大的公共领域书籍来增加训练数据大小。

这些超参数并不是为最小化损失和减少过拟合而优化的，LLM 最后生成的文本可能不是特别复杂；但这应该没关系，因为主要收获是作为速度参考的 `tok/sec` 指标（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py` 包含了下面列出的所有修改。

对比始终基于上一节第一个 epoch 后的平均 tok/秒和已分配内存。

&nbsp;
### 1. 动态创建因果掩码

- 不再保存因果掩码，而是动态创建因果掩码以减少内存使用（在这里效果很小，但在支持 131k 输入 token 的 Llama 3.2 等长上下文模型中会累积起来）

优化前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

优化后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用张量核心

- 使用张量核心（仅适用于 A100 及更新的 Ampere GPU）

优化前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

优化后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合 AdamW 优化器

- 通过设置 `fused=True` 为 `AdamW` 使用融合内核

优化前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

优化后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器中的固定内存

- 在数据加载器中使用 `pin_memory=True` 来预分配和重用 GPU 内存

优化前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

优化后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 从 32 位浮点数切换到 16 位脑浮点（bfloat16）精度（有关此主题的更多信息，请参阅我的[这篇文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

优化前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

优化后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用 PyTorch 类替换从头实现的代码

- 用 PyTorch 的原生实现替换从头实现的 LayerNorm 和 GeLU

优化前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

优化后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 的自注意力函数和 FlashAttention，而不是我们从头实现的多头注意力实现。


优化前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

优化后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。请注意，在加速之前的前几次迭代总是很慢。由于 `Avg tok/sec` 测量只包括从平均值计算的第一行，我们现在使用第 1 个 epoch 结束时的 `Step tok/sec`。


优化前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

优化后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 注意事项**

- 在 Windows 上编译可能比较棘手
- `torch.compile()` 使用 Inductor，它会即时编译内核，需要可用的 C/C++ 工具链
- 对于 CUDA，Inductor 还依赖于 Triton，可通过社区包 `triton-windows` 获取
  - 如果看到 `cl not found`，[安装带有"C++ 工作负载"的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) 并从"x64 Native Tools"提示符运行 Python
  - 如果在使用 CUDA 时看到 `triton not found`，请安装 `triton-windows`（例如，`uv pip install "triton-windows<3.4"`）
- 对于 CPU，有读者进一步建议遵循此[PyTorch Inductor Windows 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 在这里，安装 Visual Studio 2022 时安装英语语言包以避免 UTF-8 错误很重要
  - 另外，请注意代码需要通过"Visual Studio 2022 开发人员命令提示符"而不是笔记本运行
- 如果此设置证明很棘手，你可以跳过编译；**编译是可选的，所有代码示例在没有它的情况下也能正常工作**

---

&nbsp;
### 9. 词表填充

- 在这里，我们将词表大小从 50,257 稍微增加到 50,304，这是最接近的 64 的倍数。这个技巧是我的前同事 Carlos Mocholi 向我建议的，他提到这最初来自 Andrej Karpathy（可能来自[这篇帖子](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议基于与 PyTorch 团队的互动，他们给出了关于 `torch.compile` 的建议，正如 [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) 所提到的。关于这一点的一个很好的资源是 [NVIDIA 关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中批次大小和线性层维度通常被选择为某些值的倍数。此外，词表填充技巧很久以前由 NVIDIA 的 Megatron 团队描述过（参见 2019 年的 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 论文）。

优化前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

优化后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加批次大小

- 最后，我们将批次大小增加到 GPU 支持的最大 2 的幂次方

优化前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

优化后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## 多 GPU 速度对比

这可能不是完全公平的比较，因为我们现在使用 4 个 GPU 而不是 1 个，但是使用分布式数据并行（如果训练不受 GPU 内存限制，这是可以使用的最快多 GPU 技术），当然可以实现显著的加速：

优化前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

优化后（4 个 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
