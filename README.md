# GPT-Style Token Embedding Pipeline

A hands-on Jupyter Notebook that builds the **input pipeline and embedding layer** for a GPT-style language model from scratch using PyTorch and tiktoken.

---

## Overview

This notebook demonstrates the foundational steps required before training a language model:

1. Loading and reading raw text data
2. Tokenizing text using OpenAI's GPT-2 tokenizer
3. Creating a sliding-window dataset for next-token prediction
4. Building token embeddings
5. Projecting embeddings to vocabulary logits via a language model head

---

## File Structure

```
Embedding.ipynb       # Main notebook
hi_part_1.txt         # Input text file (required — place in the same directory)
```

---

## Requirements

Install dependencies before running the notebook:

```bash
pip install torch tiktoken
```

| Package    | Purpose                              |
|------------|--------------------------------------|
| `torch`    | Tensor operations, Dataset, DataLoader, Embedding, Linear layers |
| `tiktoken` | OpenAI's fast BPE tokenizer (GPT-2 encoding) |

---

## Pipeline Walkthrough

### Step 1 — Load Raw Text
Reads the first ~2000 lines from `hi_part_1.txt` into memory as a string.

### Step 2 — Tokenize
Encodes the raw text into a list of integer token IDs using the GPT-2 vocabulary (50,257 tokens).

```python
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(raw_text)
```

### Step 3 — Sliding-Window Dataset
`GPTDataset` creates overlapping input/target pairs for next-token prediction:

| Parameter    | Value | Meaning                          |
|--------------|-------|----------------------------------|
| `max_length` | 64    | Sequence length (context window) |
| `stride`     | 32    | Overlap between consecutive windows |
| `batch_size` | 8     | Samples per batch                |

For each window:
- **Input:** tokens `[i : i + 64]`
- **Target:** tokens `[i+1 : i + 65]` ← shifted by 1 (next-token prediction)

### Step 4 — Token Embedding Layer
Maps each token ID to a dense vector:

```python
embedding = torch.nn.Embedding(vocab_size=50257, embedding_dim=128)
```

Output shape: `(batch=8, seq_len=64, embed_dim=128)`

### Step 5 — Decode & Inspect
Decodes token sequences back to text and visualizes token → next-token pairs at both the text and integer-ID level.

### Step 6 — Language Model Head
A linear projection from embedding space to vocabulary logits:

```python
lm_head = torch.nn.Linear(128, 50257)
logits = lm_head(embedded_output)
# Shape: (8, 64, 50257)
```

These logits are the raw scores fed into a softmax/cross-entropy loss during training.

---

## Tensor Shape Summary

| Stage               | Shape                  |
|---------------------|------------------------|
| Raw token IDs       | `(N,)` — 1D sequence   |
| Input batch         | `(8, 64)`              |
| Embedded output     | `(8, 64, 128)`         |
| Logits              | `(8, 64, 50257)`       |
| Embedding weights   | `(50257, 128)`         |

---

## How to Run

1. Clone or download this notebook.
2. Place `hi_part_1.txt` in the same directory as the notebook.
3. Install dependencies:
   ```bash
   pip install torch tiktoken
   ```
4. Open `Embedding.ipynb` in Jupyter or VS Code and run all cells top to bottom.

---

## Concepts Covered

- **BPE Tokenization** — Byte Pair Encoding via tiktoken
- **Sliding-Window Sampling** — Overlapping context windows for language modeling
- **Token Embeddings** — `nn.Embedding` as a learned lookup table
- **Next-Token Prediction** — Autoregressive training objective
- **Language Model Head** — Linear projection to vocabulary distribution

---

## Learning Goals

By the end of this notebook you will understand:
- How raw text is transformed into tensors suitable for a transformer model
- What token embeddings are and how they are learned
- How input/target pairs are constructed for causal language modeling
- The role of the LM head in producing next-token predictions

---

## Author Notes

This notebook is designed as an **educational reference** for understanding the data pipeline of GPT-style models. It intentionally avoids a full training loop to keep focus on the embedding and tokenization stages.
