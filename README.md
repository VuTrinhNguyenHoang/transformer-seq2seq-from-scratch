# Transformer Seq2Seq Translation

Implementation of a Transformer Sequence-to-Sequence translation model from scratch using PyTorch.

## Overview

This project focuses on building a machine translation model using the Transformer architecture for English-Vietnamese translation. The model is built entirely from scratch using PyTorch, based on the Encoder-Decoder architecture introduced in the paper "Attention is All You Need".

## Directory Structure

```
├── data/               # Directory for raw and processed data
├── models/             # Directory for trained models
├── notebooks/          # Jupyter notebooks for visualization and experiments
├── src/                # Main source code
│   ├── model/          # Transformer model definition
│   ├── data/           # Data processing and loading
│   ├── train/          # Training-related code
│   └── utils/          # Utility functions
└── tests/              # Unit tests
```

## Installation

```bash
# Clone repository
git clone https://github.com/VuTrinhNguyenHoang/transformer-seq2seq-from-scratch.git
cd transformer-seq2seq-from-scratch

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Prepare data:
```bash
python src/data/prepare_data.py --output_dir data --vocab_size 16000 --max_seq_length 128
```

2. Train the model:
```bash
python src/train/train.py --data_dir data --output_dir models --d_model 128 --num_heads 2 --num_layers 2 --d_ff 512 --epochs 10
```

3. Evaluate the model:
```bash
python src/train/evaluate.py --model_path models/best_model.pt --data_dir data
```

4. Generate translations:
```bash
python src/model/generate.py --model_path models/best_model.pt --tokenizer_path data --prompt "Hello, how are you?"
```

## Key Components

- Encoder-Decoder Architecture
- Multi-Head Self-Attention
- Cross-Attention
- Position-wise Feed-Forward Networks
- Positional Encoding
- Layer Normalization
- Residual Connections
- Dropout
- SentencePiece Tokenization

## References

- [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [DataCamp Tutorial: Building a Transformer with PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)
