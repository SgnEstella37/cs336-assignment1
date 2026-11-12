# CS336 Assignment 1: Basics

本项目基于 [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 的 Assignment 1（Basics），按课程要求完成了 BPE Tokenizer、Transformer Language Model、优化器与训练相关组件的实现，并在 TinyStories 上进行了训练与文本生成实验。

详细作业说明见课程官方 Assignment 1 仓库与 handout：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。

## 核心实现

按 Assignment 1 要求实现的主要模块如下（测试通过 [`tests/adapters.py`](tests/adapters.py) 接入）：

| 模块 | 说明 | 代码 |
|------|------|------|
| BPE Tokenizer 训练 | 基于 GPT-2 风格正则预分词的 BPE 训练 | [`cs336_basics/tokenization/bpe.py`](cs336_basics/tokenization/bpe.py) |
| Tokenizer | `encode` / `decode`、特殊 token、从文件加载 | [`cs336_basics/tokenization/tokenizer.py`](cs336_basics/tokenization/tokenizer.py) |
| Transformer LM | Embedding、多层 Transformer Block、输出 LM Head | [`cs336_basics/model/transformer.py`](cs336_basics/model/transformer.py) |
| Multi-Head Self-Attention | Scaled Dot-Product Attention + 因果掩码 | [`cs336_basics/model/attention.py`](cs336_basics/model/attention.py) |
| RoPE / RMSNorm / Softmax / Linear / Embedding | 位置编码与基础层 | [`cs336_basics/model/attention.py`](cs336_basics/model/attention.py)、[`cs336_basics/model/layers.py`](cs336_basics/model/layers.py) |
| SwiGLU | Feed-Forward 网络 | [`cs336_basics/model/feedforward.py`](cs336_basics/model/feedforward.py) |
| AdamW | 优化器 | [`cs336_basics/training/AdamW_optimizer.py`](cs336_basics/training/AdamW_optimizer.py) |
| 训练配套 | Cross-Entropy、Cosine LR Schedule、Gradient Clipping、Data Loader、Checkpoint | [`cs336_basics/training/`](cs336_basics/training/) |
| Training Loop | 训练 / 验证、W&B 日志、定期保存 checkpoint | [`cs336_basics/training/training.py`](cs336_basics/training/training.py) |
| Text Generation | Temperature / Top-p（nucleus）采样 | [`cs336_basics/generation/generation.py`](cs336_basics/generation/generation.py) |

## 训练与实验

当前仓库中的实验脚本主要围绕 **TinyStories**，面向 Apple Silicon（`device="mps"`）。定量训练曲线与 checkpoint 未纳入版本控制（见 `.gitignore` 中的 `/data/`、`/wandb/`）。

### 数据集

- TinyStories（`TinyStoriesV2-GPT4-train.txt` / `valid.txt`）
- 先训练 vocab size = 10000 的 BPE Tokenizer，再将语料编码为 `.npy` 供训练使用

相关脚本：

- Tokenizer 训练：[`cs336_basics/experiments/tinystories_related/train_tokenizer_using_TinyStories.py`](cs336_basics/experiments/tinystories_related/train_tokenizer_using_TinyStories.py)
- 数据集编码：[`cs336_basics/experiments/tinystories_related/tokenize_dataset.py`](cs336_basics/experiments/tinystories_related/tokenize_dataset.py)

### 模型与训练配置（Baseline）

脚本：[`cs336_basics/experiments/tinystories_related/controlled_experiments_ts/train_tinystories_mps_baseline.py`](cs336_basics/experiments/tinystories_related/controlled_experiments_ts/train_tinystories_mps_baseline.py)

| 超参数 | 取值 |
|--------|------|
| `vocab_size` | 10000 |
| `context_length` | 256 |
| `d_model` | 512 |
| `num_layers` | 4 |
| `num_heads` | 16 |
| `d_ff` | 1344 |
| `batch_size` | 32 |
| `max_iters` | 5000 |
| `learning_rate` | 3e-4（warmup 100，cosine 至 3e-5） |
| `device` | `mps` |
| 日志 | Weights & Biases（`project=cs336-assignment1`） |
| checkpoint | `data/tinystories_mps_checkpoint.pt` |

另提供短程冒烟测试脚本（`max_iters=10`）：[`train_tinystories_mps_smoke_test.py`](cs336_basics/experiments/tinystories_related/controlled_experiments_ts/train_tinystories_mps_smoke_test.py)。

### 文本生成

加载上述 TinyStories tokenizer 与 checkpoint，交互式生成故事：[`cs336_basics/generation/generate_tinystories.py`](cs336_basics/generation/generate_tinystories.py)（默认 `temperature=0.8`，`top_p=0.9`）。

## 项目结构

```text
cs336_basics/
  tokenization/   # BPE 训练与 Tokenizer
  model/          # Transformer LM 及基础组件
  training/       # 优化器、损失、数据加载、训练循环
  generation/     # 文本生成
  experiments/    # TinyStories 相关训练与实验脚本
tests/            # 课程提供的单元测试与 adapters
```

## 环境配置与运行

### 环境

使用 [`uv`](https://github.com/astral-sh/uv#installation) 管理依赖（Python `>=3.12,<3.14`）：

```sh
uv sync
uv run <python_file_path>
```

### 单元测试

```sh
uv run pytest
```

实现通过 [`tests/adapters.py`](tests/adapters.py) 与测试相连。

### 下载数据

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### TinyStories 训练流程示例

```sh
# 1. 训练 BPE Tokenizer
uv run cs336_basics/experiments/tinystories_related/train_tokenizer_using_TinyStories.py

# 2. 将语料编码为 .npy
uv run cs336_basics/experiments/tinystories_related/tokenize_dataset.py

# 3. 冒烟测试 / Baseline 训练（需 MPS；会写入 W&B）
uv run cs336_basics/experiments/tinystories_related/controlled_experiments_ts/train_tinystories_mps_smoke_test.py
uv run cs336_basics/experiments/tinystories_related/controlled_experiments_ts/train_tinystories_mps_baseline.py

# 4. 文本生成（需已有 tokenizer 与 checkpoint）
uv run cs336_basics/generation/generate_tinystories.py
```

## 参考资料

- 课程主页：[CS336 Language Modeling from Scratch](https://cs336.stanford.edu/)
- Assignment 1 官方仓库：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

本项目为按上述课程 Assignment 1 要求完成的作业实现。
