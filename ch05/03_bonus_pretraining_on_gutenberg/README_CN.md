# 在古腾堡计划数据集上预训练 GPT 模型

本目录包含用于在古腾堡计划（Project Gutenberg）提供的免费图书上训练小型 GPT 模型的代码。

正如古腾堡计划网站所述："古腾堡计划的绝大多数电子图书在美国属于公共领域。"

请参阅 [古腾堡计划许可、授权及其他常见请求](https://www.gutenberg.org/policy/permission.html) 页面，了解关于使用古腾堡计划所提供资源的更多信息。

&nbsp;
## 代码使用指南

&nbsp;

### 1) 下载数据集

本节中，我们使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 仓库中的代码从古腾堡计划下载图书数据。

截至本文撰写时，此过程约需 50 GB 磁盘空间，耗时约 10-15 小时，具体时间可能取决于古腾堡计划的后续增长情况。

&nbsp;
#### Linux 和 macOS 用户下载指南


Linux 和 macOS 用户可按以下步骤下载数据集（Windows 用户请参阅下方说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设为工作目录，以便在该文件夹中本地克隆 `gutenberg` 仓库（这是运行提供的 `prepare_dataset.py` 和 `pretraining_simple.py` 脚本的必要步骤）。例如，当处于 `LLMs-from-scratch` 仓库文件夹时，可通过以下命令进入 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在该目录下克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 仓库文件夹：
```bash
cd gutenberg
```

4. 从 `gutenberg` 仓库文件夹安装 *requirements.txt* 中定义的依赖包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回 `03_bonus_pretraining_on_gutenberg` 文件夹：
```bash
cd ..
```

&nbsp;
#### Windows 用户特别说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码兼容 Linux 和 macOS 系统。但 Windows 用户需要进行少量适配，例如在 `subprocess` 调用中添加 `shell=True` 参数以及替换 `rsync` 命令。

另一种在 Windows 上运行此代码的更简便方式是使用"适用于 Linux 的 Windows 子系统"（WSL）功能，该功能允许用户在 Windows 环境中运行基于 Ubuntu 的 Linux 系统。详情请参阅 [微软官方安装指南](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保已安装 Python 3（通过 `python3 --version` 检查，或使用 `sudo apt-get install -y python3.10` 安装 Python 3.10），并安装以下软件包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注：**
> 关于 Python 环境配置和软件包安装的说明，可参阅 [可选 Python 配置偏好](../../setup/01_optional-python-setup-preferences/README.md) 和 [安装 Python 库](../../setup/02_installing-python-libraries/README.md)。
>
> 此外，本仓库还提供了运行 Ubuntu 的 Docker 镜像。关于使用所提供 Docker 镜像运行容器的说明，可参阅 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2) 准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至本文撰写时共 60,173 个）文本文件合并为数量更少、体积更大的文件，以便更高效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，生成的文件以纯文本格式存储，为简洁起见未进行预分词处理。但若您计划频繁使用该数据集或进行多轮训练，建议修改代码将数据集以预分词形式存储，以节省计算时间。详情请参阅本页底部的 *设计决策与改进建议*。

> **提示：**
> 您可选择更小的文件大小，例如 50 MB。这将产生更多文件，但便于在少量文件上进行快速预测试运行。


&nbsp;
### 3) 运行预训练脚本

可按以下方式运行预训练脚本。注意：为说明起见，此处显示了带有默认值的额外命令行参数：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出格式如下所示：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **提示：**
> 实际应用中，若您使用 macOS 或 Linux 系统，建议使用 `tee` 命令将日志输出同时保存到 `log.txt` 文件并显示在终端上：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 请注意，在 V100 GPU 上对 `gutenberg_preprocessed` 文件夹中单个约 500 MB 的文本文件进行训练约需 4 小时。
> 该文件夹包含 47 个文件，全部完成约需 200 小时（超过 1 周）。建议您先在少量文件上进行测试。


&nbsp;
## 设计决策与改进建议

请注意，本代码以教学为目的，力求保持简洁精简。可通过以下方式改进代码，以提升建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本，从各图书文件中剔除古腾堡计划的标准说明文本。
2. 更新数据准备和加载工具，对数据集进行预分词并以分词形式保存，避免每次调用预训练脚本时重新分词。
3. 通过添加 [附录 D：训练循环增强功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的功能来更新 `train_model_simple` 脚本，包括余弦衰减、线性预热和梯度裁剪。
4. 更新预训练脚本以保存优化器状态（参见第 5 章中的 *5.4 PyTorch 中的权重加载与保存* 章节；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加加载现有模型和优化器检查点以及在训练中断后继续训练的选项。
5. 添加更高级的日志记录工具（如 Weights and Biases），以实时查看损失和验证曲线。
6. 添加分布式数据并行（DDP）功能，在多个 GPU 上训练模型（参见附录 A 中的 *A.9.3 多 GPU 训练* 章节；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中从零实现的 `MultiheadAttention` 类替换为 [高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 附加章节中实现的高效 `MHAPyTorchScaledDotProduct` 类，该类通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention 技术。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) 或 [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`) 优化模型以加速训练。
9. 实现梯度低秩投影（GaLore）技术以进一步加速预训练过程。只需将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中提供的 `GaLoreAdamW` 即可实现。
