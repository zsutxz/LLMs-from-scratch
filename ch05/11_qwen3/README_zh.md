# 从零实现 Qwen3

本文件夹中的 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter notebook 包含了 Qwen3 0.6B、1.7B、4B、8B 和 32B 的从零实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


本文件夹中的 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) Jupyter notebook 包含了 30B-A3B 混合专家（MoE）模型的从零实现，包括 Thinking、Instruct 和 Coder 模型变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Qwen3 从零代码

本文件夹中的独立 notebook 包含以线性方式编写的从零代码：

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb)：不带花哨功能的密集 Qwen3 模型
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb)：与上述相同，但带有 KV 缓存以获得更好的推理效率
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)：类似第一个 notebook，但是混合专家（MoE）变体
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb)：与上述相同，但带有 KV 缓存以获得更好的推理效率

或者，我还在[这里](../../pkg/llms_from_scratch/)将代码组织成了一个 Python 包（包括单元测试和 CI），你可以按照下面的描述运行它。

&nbsp;
# 训练

`Qwen3Model` 类的实现风格与 `GPTModel` 类类似，因此它可以用作第 5 章训练和第 6、7 章微训的直接替代品。


&nbsp;
# 通过 `llms-from-scratch` 包使用 Qwen3

为了更方便地使用 Qwen3 的从零实现，你也可以使用基于本仓库 [pkg/llms_from_scratch](../../pkg/llms_from_scratch) 中源代码的 `llms-from-scratch` PyPI 包。

&nbsp;
#### 1) 安装

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) 模型和文本生成设置

指定要使用的模型：

```python
USE_REASONING_MODEL = True
# 如果 USE_REASONING_MODEL = False，则使用基础模型

USE_INSTRUCT_MODEL = False
# 如果以下条件为真，则使用指令模式（不带推理）：
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# 如果 USE_REASONING_MODEL = False，此设置无效


# 使用
# USE_REASONING_MODEL = True
# 同样适用于 Qwen3 Coder Flash 模型
```

用户可以定义的基本文本生成设置。对于 150 个 token，0.6B 模型需要大约 1.5 GB 内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) 0.6B 模型的权重下载和加载

以下代码根据上述模型选择（推理或基础）自动下载权重文件。请注意，本节重点关注 0.6B 模型。如果你想使用任何更大的模型（1.7B、4B、8B 或 32B），请跳过本节并继续阅读第 3b 节。

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"
else:
    filename = "qwen3-0.6B-base.pth"
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后按如下方式加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) 更大 Qwen 模型的权重下载和加载

如果你有兴趣使用任何更大的 Qwen 模型，例如 1.7B、4B、8B 或 32B，请使用以下代码代替 3a 节下的代码，这需要额外的代码依赖：

```bash
pip install safetensors huggingface_hub
```

然后使用以下代码（对 `USE_MODEL` 进行适当更改以选择所需的模型大小）

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")

repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，下载权重并将其加载到 `model` 中：

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # 仅对 MoE 模型需要
del weights_dict  # 删除权重字典以释放磁盘空间
```



&nbsp;

#### 4) 初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"
else:
    tok_filename = "tokenizer-base.json"

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



&nbsp;

#### 5) 生成文本

最后，我们可以通过以下代码生成文本：

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

使用 Qwen3 0.6B 推理模型时，输出应该类似于下面显示的结果（这是在 A100 上运行的）：

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



对于更大的模型，你可能更喜欢流式变体，它在生成每个 token 时立即打印：

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



&nbsp;

#### 专业技巧 1：通过编译加速推理


要获得高达 4 倍的加速，请将

```python
model.to(device)
```

替换为

```python
model.to(device)
model = torch.compile(model)
```

注意：编译时有显著的数分钟前期成本，加速在第一次 `generate` 调用后生效。

下表显示了在 A100 上连续 `generate` 调用的性能比较：

|                          | Hardware        | Tokens/sec | Memory   |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B compiled | Nvidia A100 GPU | 107        | 1.99 GB  |


&nbsp;
#### 专业技巧 2：通过 KV 缓存加速推理

在 CPU 上运行模型时，你可以通过使用 KV 缓存 `Qwen3Model` 直接替代品来显著提升推理性能。（要了解更多关于 KV 缓存的信息，请参阅我的文章《[从零理解和编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)》。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

请注意，峰值内存使用量仅针对 Nvidia CUDA 设备列出，因为它更容易计算。然而，其他设备上的内存使用量可能类似，因为它使用类似的精度格式，并且 KV 缓存存储在这里生成的 150 token 文本的内存使用量甚至更低（但是，不同设备可能以不同方式实现矩阵乘法，可能导致不同的峰值内存需求；对于更长的上下文长度，KV-cache 内存可能会增加得无法接受）。

| Model           | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| --------------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 CPU | 137        | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 GPU | 21         | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 GPU | 28         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | Regular compiled  | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV cache          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV cache compiled | Nvidia A100 GPU | 90         | 1.48 GB           |

请注意，上述所有设置都已测试以产生相同的文本输出。



&nbsp;

#### 专业技巧 3：批量推理

我们可以通过批量推理进一步增加吞吐量。虽然这不是同类比较，因为我们要用更多的输入序列运行推理，但这会增加每秒 token 的吞吐量，同时以增加内存使用为代价。

这只需要对提示准备进行小的代码修改。例如，考虑下面的批量提示：

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存版本的代码类似，只是需要使用这些直接替代品：

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


以下实验的批次大小为 8。

| Model            | Mode              | Hardware        | Batch size | Tokens/sec | GPU Memory (VRAM) |
| ---------------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model  0.6B | Regular           | Mac Mini M4 CPU | 8          | 2          | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 CPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 CPU | 8          | 92         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 CPU | 8          | 128        | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Mac Mini M4 GPU | 8          | 36         | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 GPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 GPU | 8          | 61         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 GPU | 8          | -          | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | Regular compiled  | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV cache          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV cache compiled | Nvidia A100 GPU | 8          | 280        | 1.75 GB           |
