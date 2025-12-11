# CLAUDE 技术指南

**原文链接**: E:\AI\LLMs-from-scratch\CLAUDE.md
**翻译时间**: 2025年11月25日
**文章类型**: 技术文档

---

本文件为 Claude Code (claude.ai/code) 在处理此代码仓库时提供指导。

## 项目概述

这是 Sebastian Raschka 所著《从零构建大语言模型》(Build a Large Language Model (From Scratch)) 一书的官方代码仓库。它包含使用 PyTorch 从零开始的类 GPT 模型的教育实现，按章节组织并提供逐步解释。

## 开发环境设置

### 快速安装
```bash
pip install -r requirements.txt
```

### 环境管理器
- **UV (推荐)**: 使用 `uv pip install -r requirements.txt` 或 `uv pip install --group bonus` 安装所有依赖
- **Pixi**: 使用 `pixi shell` 激活环境（在 pixi.toml 中配置）
- **Docker**: 在 `setup/03_optional-docker-environment` 中提供 DevContainer 设置

### 核心依赖
- PyTorch（版本因平台和 Python 版本而异，详见 pyproject.toml）
- JupyterLab（用于执行 Jupyter 笔记本）
- Tiktoken（用于分词处理）
- Matplotlib（用于数据可视化）
- TensorFlow（某些章节需要）
- NumPy, Pandas, TQDM

## 项目结构

### 章节组织
- `ch01-ch07/`: 主要章节内容，包括：
  - `01_main-chapter-code/`: 主要代码示例和笔记本
  - 编号子文件夹中的附加材料（例如 `02_bonus_*`）
  - 每章都包含 `exercise-solutions.ipynb` 中的练习解答

### 按章节分类的关键文件
- **第2章**: 文本数据处理和分词
- **第3章**: 注意力机制实现
- **第4章**: GPT 模型架构（`gpt.py`）
- **第5章**: 预训练脚本（`gpt_train.py`, `gpt_generate.py`）
- **第6章**: 分类微调（`gpt_class_finetune.py`）
- **第7章**: 指令微调（`gpt_instruction_finetuning.py`）

### 附加资源
- `appendix-A/`: PyTorch 介绍和 DDP（分布式数据并行）示例
- `appendix-D/`: 训练循环增强功能
- `appendix-E/`: LoRA（低秩适应）实现
- `setup/`: 环境设置指南
- `reasoning-from-scratch/`: 高级推理模型（配套书籍）

## 测试和验证

### 运行测试
```bash
# 使用 pytest
pytest

# 使用 UV（如果已安装）
uv run pytest
```

### Jupyter 笔记本测试
某些工作流使用 `nbval` 测试 Jupyter 笔记本：
```bash
pytest --nbval path/to/notebook.ipynb
```

## 代码质量

### 代码检查和格式化
- **Ruff**: 配置行长度为 140，在 pyproject.toml 中有特定的忽略规则
- 代码检查器排除 `.venv` 虚拟环境目录
- 通过 GitHub Actions 或本地运行 `ruff check` 进行代码检查

## 常见开发任务

### 运行单个章节代码
```bash
# 导航到章节目录
cd ch02/01_main-chapter-code/

# 启动 JupyterLab
jupyter lab

# 或直接运行 Python 脚本
python script_name.py
```

### 使用附加材料
许多章节包含丰富的附加材料：
- 替代注意力机制（第4章）
- 额外的模型架构（第5章）
- 数据集工具和评估方法（第6-7章）
- 用户界面示例

### GPU 支持
代码在可用时自动检测并使用 GPU。基本使用无需手动设备管理。

## 包开发

本仓库也可作为 PyPI 包安装：
```bash
pip install -e .
```

包源代码位于 `pkg/` 目录中，在 `pyproject.toml` 中配置。

## 平台特定说明

### Windows
- 自动使用 TensorFlow CPU 版本
- 为兼容性使用特定的 PyTorch 版本

### macOS
- Intel 和 Apple Silicon 有单独的依赖版本
- TensorFlow 版本因架构而异

### GPU 支持
- CUDA 检测是自动的
- 如果 GPU 不可用则回退到 CPU
- 第5-7章最受益于 GPU 加速

## 推荐工作流程

1. **环境设置**: 使用首选方法（UV/pixi/pip）安装依赖
2. **章节导航**: 按顺序学习章节，从 ch02 开始
3. **代码执行**: 使用 JupyterLab 进行交互式开发
4. **附加内容**: 在完成主要章节后探索附加材料
5. **练习**: 在查看解答前尝试练习

## 重要说明

- 所有主要章节代码都设计为可在普通笔记本电脑上运行，无需专业硬件
- GPU 加速在可用时自动启用
- 附加材料可能需要额外依赖（使用 `uv pip install --group bonus` 安装）
- 仓库保持与实体书的一致性 - 不接受会改变主要章节内容的贡献
- 如有问题，请使用 Manning 论坛或 GitHub Discussions