# Large Language Model — From-Scratch GPT Implementation & Fine-Tuning Framework

This repository is a complete, portfolio-grade implementation of a **GPT-style Large Language Model**, built entirely from scratch using PyTorch.
It includes:

- A fully custom Transformer architecture (embeddings → multi-head attention → feedforward → layer norm → residuals).

- A flexible training pipeline for both language modeling and downstream tasks.

- A spam classifier fine-tuned from a GPT model.

- A small instruction-tuned personal assistant model trained on a curated instruction dataset.

- Optional support for loading GPT-2 weights from TensorFlow checkpoints.

The project demonstrates end-to-end understanding of LLMs: architecture, data loading, training loops, fine-tuning, inference, and evaluation.

This repository is designed as a personal ML engineering showcase, emphasizing clarity, reproducibility, and modular design.

## Installation

### 1. Clone the Repository
```
git clone https://github.com/Hrishi1084/Large-Language-Model
cd Large-Language-Model
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Install the Package (Editable Mode)
```
pip install -e .
```

## Key Features
**1. Fully Custom Transformer + GPT Implementation**

- Token embeddings

- Positional encodings
- Multi-Head Self-Attention
- Feedforward / MLP
- Residual connections
- Layer normalization
- Next-token prediction head

Implemented across:

- `attention_mechanisms.ipynb`

- `transformer_block.ipynb`

- `large_language_model.ipynb`

- `all_modules.py`

**2. Complete Training Stack**

- Custom dataset loaders
- Tokenization utilities
- Batch samplers & collators
- Training loop with loss tracking
- Text generation & sampling functions

**3. GPT-2 Weight Import Support**

Download and port GPT-2 TensorFlow checkpoints via:

- `gpt_download.py` 

- Custom checkpoint loader → PyTorch compatibility

**4. Downstream Fine-Tuning**

Two full pipelines:

**a. Spam Classifier Fine-Tuning**

- Fine-tunes GPT to classify SMS spam.
- Uses train/validation/test CSVs.
- Implementation: `finetuned_classifier.ipynb` 

**b. Instruction-Tuned Personal Assistant**

A small SFT-style model trained on curated instruction–response pairs:

- Dataset: `instruction-data.json` 

- Training script: `finetuned_personal_assistant.ipynb` 

- Evaluation: `the-verdict.txt`

**5. Recruiter-Friendly Code Quality**

- Modular structure

- Clear component boundaries

- Readable training loops

- Extensible and documented code design

- Reproducible pipelines

## Project Structure
```
Large-Language-Model/
│
├── data/
│ ├── train.csv
│ ├── validation.csv
│ ├── test.csv
│ ├── the-verdict.txt
│ └── sms_spam_collection/
│
├── models/
│ ├── gpt2/
│ ├── gpt2-medium355M-sft.pth
│ ├── model.pth
│ └── review_classifier.pth
│
├── notebooks/
│ ├── attention_mechanisms.ipynb
│ ├── evaluate_personal_assistant.ipynb
│ ├── finetuned_classifier.ipynb
│ ├── finetuned_personal_assistant.ipynb
│ ├── input_embeddings_generator.ipynb
│ ├── large_language_model.ipynb
│ ├── load_pretrained_model.ipynb
│ └── transformer_block.ipynb
│
├── plots/
│ ├── accuracy-plot.pdf
│ ├── loss-plot.pdf
│ ├── new-loss-plot.pdf
│ └── temperature_plot.pdf
│
├── src/
│ └── large_language_model/
│     ├── init.py
│     ├── all_modules.py
│     ├── attention_mechanisms.py
│     ├── finetuned_classifier.py
│     ├── finetuned_personal_assistant.py
│     ├── gpt_download.py
│     ├── large_language_model.py
│     └── transformer_block.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Key Modules Explained

`attention_mechanisms.ipynb` : Implements multi-head self-attention, including Q/K/V projections, attention weights, and head merging.

`transformer_block.ipynb` : Defines a full Transformer decoder block: attention → MLP → residuals → layer norm.

`all_modules.py` : A collection of reusable model components: embeddings, feedforward layers, layer norms, and utility modules.

`large_language_model.ipynb` : Combines all components into a complete GPT-style autoregressive model.

`gpt_download.py` : Downloads GPT-2 TensorFlow weights and converts them into a structured Python dict for reuse. 

`finetuned_classifier.ipynb` : Full pipeline for GPT-based spam classification. 

`finetuned_personal_assistant.ipynb` : Training logic for instruction-tuning on curated JSON examples. 


## Model Architecture Overview

GPT implementation follows the standard decoder-only transformer architecture:
```
Input Tokens
    ↓
Token Embeddings
    ↓
Positional Embeddings
    ↓
────────────────────────────
│ N × Transformer Blocks   │
│                          │
│  ┌───────────┐           │
│  │ LayerNorm │           │
│  ├───────────┤           │
│  │ Self-Attn │           │
│  └───────────┘           │
│        ↓                 │
│   Residual Add           │
│        ↓                 │
│  ┌───────────┐           │
│  │ LayerNorm │           │
│  ├───────────┤           │
│  │   MLP     │           │
│  └───────────┘           │
│        ↓                 │
│   Residual Add           │
────────────────────────────
    ↓
Linear LM Head
    ↓
Softmax (next-token probabilities)
```

## Key design capabilities

- Works for pretraining-style next-token prediction.

- Easily adapts to classification via a sequence representation head.

- Supports instruction tuning through next-token loss on formatted (instruction, input, output) pairs.

