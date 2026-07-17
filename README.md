# Build DeepSeek From Scratch

This repository contains my implementations and study notes for building the main
components of a modern large language model from first principles, with a focus on the
design choices behind the DeepSeek models. Each topic is developed in a Jupyter notebook
that combines the underlying mathematics, a PyTorch implementation, and visualizations of
the intermediate tensors.

The aim is to understand how a large language model works by reconstructing it one
component at a time, rather than relying on a high level library that hides the details.

## Overview

Large language models are assembled from a compact set of ideas: attention, positional
encoding, mixture of experts, and inference time optimizations such as key value caching
and quantization. This repository reimplements each of these in isolation and then
combines them to reproduce the architecture used in the DeepSeek family of models. Every
notebook is written to be read top to bottom, with the mathematics introduced just before
the code that implements it.

## Notebooks in this repository

| Notebook | Topic |
| - | - |
| 01. Self_Attention_Mechanism.ipynb | Self attention with trainable query, key, and value projections |
| 02. Causal_Attention.ipynb | Causal masking, attention dropout, and a compact batched attention module |
| 03. Multi_Head_Attention.ipynb | Multi head attention |

## Topics

The material is organized into six modules.

### 1. Foundations
* Overview of the language model architecture

### 2. Attention
* Self attention
* Causal (masked) attention
* Multi head attention
* Handwritten notes on multi head attention
* Key value (KV) cache
* Multi query attention (MQA)
* Grouped query attention (GQA)
* Multi head latent attention (MLA)
* The attention mechanism used in DeepSeek

### 3. Positional encoding
* Integer and binary positional encoding
* Sinusoidal positional encoding
* Rotary positional encoding (RoPE)

### 4. Mixture of experts
* Introduction to the mixture of experts layer
* A hands on mixture of experts demonstration
* Load balancing techniques for mixture of experts
* DeepSeekMoE
* A mixture of experts layer implemented from scratch

### 5. Multi token prediction
* Introduction to multi token prediction
* Multi token prediction in DeepSeek
* Multi token prediction implemented from scratch

### 6. Quantization
* Quantization for language models
* Quantization techniques, part one and part two
* Quantization implemented from scratch

## Getting started

Clone the repository and create a virtual environment. Python 3.13 is recommended.

```bash
git clone https://github.com/RajiaRani/Build_DeepSeek_From_Scratch.git
cd Build_DeepSeek_From_Scratch

python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter and select the `.venv` interpreter as the
kernel. The virtual environment is excluded from version control because it is large and
can be recreated from `requirements.txt`.

## Requirements

The notebooks depend on PyTorch for the model code and NumPy and Matplotlib for the
mathematics and visualizations. Exact versions are pinned in `requirements.txt`.

## References

* Vaswani et al. Attention Is All You Need. 2017.
* Shazeer. Fast Transformer Decoding: One Write Head is All You Need. 2019.
* Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021.
* Ainslie et al. GQA: Training Generalized Multi Query Transformer Models from Multi Head Checkpoints. 2023.
* DeepSeek AI. DeepSeek V2: A Strong, Economical, and Efficient Mixture of Experts Language Model. 2024.
* DeepSeek AI. DeepSeek V3 Technical Report. 2024.
