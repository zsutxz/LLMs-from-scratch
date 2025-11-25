# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official code repository for the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka. It contains educational implementations of GPT-like models from scratch using PyTorch, organized by chapter with step-by-step explanations.

## Development Environment Setup

### Quick Installation
```bash
pip install -r requirements.txt
```

### Environment Managers
- **UV (recommended)**: Use `uv pip install -r requirements.txt` or `uv pip install --group bonus` for all dependencies
- **Pixi**: Use `pixi shell` to activate the environment (configured in pixi.toml)
- **Docker**: DevContainer setup available in `setup/03_optional-docker-environment`

### Core Dependencies
- PyTorch (version varies by platform and Python version, see pyproject.toml)
- JupyterLab (for notebook execution)
- Tiktoken (for tokenization)
- Matplotlib (for plotting)
- TensorFlow (required for certain chapters)
- NumPy, Pandas, TQDM

## Project Structure

### Chapter Organization
- `ch01-ch07/`: Main chapter content with:
  - `01_main-chapter-code/`: Primary code examples and notebooks
  - Bonus materials in numbered subfolders (e.g., `02_bonus_*`)
  - Each chapter includes exercise solutions in `exercise-solutions.ipynb`

### Key Files by Chapter
- **Chapter 2**: Text data processing and tokenization
- **Chapter 3**: Attention mechanism implementations
- **Chapter 4**: GPT model architecture (`gpt.py`)
- **Chapter 5**: Pretraining scripts (`gpt_train.py`, `gpt_generate.py`)
- **Chapter 6**: Classification finetuning (`gpt_class_finetune.py`)
- **Chapter 7**: Instruction finetuning (`gpt_instruction_finetuning.py`)

### Additional Resources
- `appendix-A/`: PyTorch introduction and DDP examples
- `appendix-D/`: Training loop enhancements
- `appendix-E/`: LoRA implementation
- `setup/`: Environment setup guides
- `reasoning-from-scratch/`: Advanced reasoning models (companion book)

## Testing and Validation

### Running Tests
```bash
# Using pytest
pytest

# Using UV (if installed)
uv run pytest
```

### Notebook Testing
Some workflows use `nbval` for testing Jupyter notebooks:
```bash
pytest --nbval path/to/notebook.ipynb
```

## Code Quality

### Linting and Formatting
- **Ruff**: Configured with line length 140, specific ignore rules in pyproject.toml
- Linter excludes `.venv` directories
- Run linting via GitHub Actions or locally with `ruff check`

## Common Development Tasks

### Running Individual Chapter Code
```bash
# Navigate to chapter directory
cd ch02/01_main-chapter-code/

# Start JupyterLab
jupyter lab

# Or run Python scripts directly
python script_name.py
```

### Working with Bonus Materials
Many chapters contain extensive bonus materials:
- Alternative attention mechanisms (Chapter 4)
- Additional model architectures (Chapter 5)
- Dataset utilities and evaluation methods (Chapters 6-7)
- User interface examples

### GPU Support
Code automatically detects and uses GPUs when available. No manual device management required for basic usage.

## Package Development

This repository is also installable as a PyPI package:
```bash
pip install -e .
```

The package source code is in the `pkg/` directory, configured in `pyproject.toml`.

## Platform-Specific Notes

### Windows
- TensorFlow CPU version is used automatically
- Specific PyTorch versions for compatibility

### macOS
- Intel and Apple Silicon have separate dependency versions
- TensorFlow versions differ by architecture

### GPU Support
- CUDA detection is automatic
- Fallback to CPU if GPU unavailable
- Chapters 5-7 benefit most from GPU acceleration

## Recommended Workflow

1. **Setup**: Install dependencies using preferred method (UV/pixi/pip)
2. **Chapter Navigation**: Work through chapters sequentially, starting with ch02
3. **Code Execution**: Use JupyterLab for interactive development
4. **Bonus Content**: Explore bonus materials after main chapter completion
5. **Exercises**: Attempt exercises before checking solutions

## Important Notes

- All main chapter code is designed to run on conventional laptops without specialized hardware
- GPU acceleration is automatic when available
- Bonus materials may require additional dependencies (install with `uv pip install --group bonus`)
- Repository maintains consistency with the physical book - contributions that would change main chapter content are not accepted
- For questions, use Manning Forum or GitHub Discussions