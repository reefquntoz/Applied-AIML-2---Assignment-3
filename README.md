# Transformer From Scratch for IMDb Sentiment Analysis — Assignment_3_v1-Arif-a1915248

This project implements a simplified Transformer encoder from scratch in PyTorch to classify IMDb movie reviews.

This README uses `Assignment_3_v1-Arif-a1915248.ipynb` as the reference notebook for the global settings, with a **Further Variants** section covering the other 7 notebooks.

## Summary

`Assignment_3_v1-Arif-a1915248.ipynb` tokenises IMDb reviews with a custom word-frequency vocabulary (20,000 words), builds a 4-layer Transformer encoder (`d_model=64`, 4 attention heads, feed-forward width 256), and trains it for 10 epochs with the Adam optimiser (`lr=0.001`, batch size 32) and binary cross-entropy loss.

## Project Files

- `Assignment_3_v1-Arif-a1915248.ipynb` — the notebook (dataset loading, EDA, tokenisation, training/evaluation calls).
- `train_utils.py` — Defines `train()` and `test()`, the shared training/evaluation loops used by every notebook version.
- `transformer_encoder.py` — Defines the shared `TransformerEncoder` model and its building blocks (`TransformerEncoderBlock`, `MultiHeadAttention`, `PositionalEncoding`, `scaled_dot_product_attention`).
- `tokenisation_utils.py` — Defines the shared word-frequency tokenisation functions (`tokenise_words`, `build_vocabulary`, `compute_oov_rate`, `tokenise_function`).

## Requirements

- Python 3.9+
- Packages:
  - `torch`
  - `datasets` (HuggingFace)
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `transformers` — only needed for `Assignment_3_v2-Arif-a1915248.ipynb` (BERT tokenizer variant)
- Internet access, to download the IMDb dataset from the HuggingFace Hub on first run.

## Dataset

- **Dataset**: IMDb movie reviews (`stanfordnlp/imdb`), loaded via `datasets.load_dataset`.
- **Split**: 25,000 train reviews and 25,000 test reviews, each split balanced 12,500 positive / 12,500 negative.
- **Tokenisation**: a custom word-frequency vocabulary is built from the training text only (never the test set). HTML `<br />` tags are stripped, text is lowercased and split into words, and only the 20,000 most frequent words are kept (plus `[PAD]` and `[UNK]`). Unknown words map to `[UNK]`. Train OOV rate: 2.29%. Test OOV rate: 2.97%.
- **Sequence length**: each review is truncated or padded to `max_seq_length = 600` tokens.

## Step-by-step: How to Run

1. Keep `Assignment_3_v1-Arif-a1915248.ipynb`, `train_utils.py`, `transformer_encoder.py`, and `tokenisation_utils.py` in the same folder.
2. Install the required packages.
3. Run all cells top to bottom:
   - Downloads the IMDb dataset (first run only, needs internet access).
   - Builds the word-frequency vocabulary, tokenises the dataset, and creates DataLoaders.
   - Builds the `TransformerEncoder`, trains it for 10 epochs, and evaluates it on the test set.

Note: a full training run takes roughly 20 minutes on a GPU.

## Model Architecture

`Assignment_3_v1-Arif-a1915248.ipynb` uses a `TransformerEncoder`:

| Stage | Details |
|---|---|
| Input | Token ids, shape (batch, 600) |
| Embedding | `torch.nn.Embedding(20000, 64)`, padding id fixed at zero |
| Positional Encoding | Fixed sinusoidal encoding, added to the embeddings |
| Encoder Block x4 | Multi-head self-attention (4 heads) -> residual + LayerNorm -> feed-forward (64 -> 256 -> 64, ReLU) -> residual + LayerNorm |
| Pooling | Mean-pool token representations, ignoring padding positions |
| Classification Head | `torch.nn.Linear(64, 2)` -> Softmax |

Implementation notes:
- Attention is standard scaled dot-product attention.
- Each encoder block is the "post-layer-norm" variant.
- The attention mask (from padding) is used both inside self-attention and during mean pooling.

## Hyperparameters

### `Assignment_3_v1-Arif-a1915248.ipynb`
| Hyperparameter | Value |
|---|---|
| Vocabulary size | 20,000 |
| Max sequence length | 600 |
| Encoder layers | 4 |
| Model dimension (d_model) | 64 |
| Attention heads | 4 |
| Feed-forward width (d_ff) | 256 |
| Dropout | 0.1 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Epochs | 10 |
| Optimiser | Adam |
| Loss function | Binary Cross-Entropy |

## Results

### `Assignment_3_v1-Arif-a1915248.ipynb`
- Final training loss (epoch 10/10): **0.2053**; final validation loss: **0.1645**
- Test set — accuracy: **0.8664**; weighted avg precision/recall/f1: **0.868 / 0.866 / 0.866**
- Training time: ~1224 s (~20.4 min)

## Further Variants

Each variant below changes exactly one setting from the `v1` baseline above; everything else stays the same.

| Notebook | Change from v1 | Test Accuracy |
|---|---|---|
| `Assignment_3_v2-Arif-a1915248.ipynb` | BERT-base-uncased tokenisation (WordPiece tokenizer, retrained on the IMDb text so the vocabulary still caps at 20,000) | 0.8596 |
| `Assignment_3_v3.1-Arif-a1915248.ipynb` | `max_seq_length = 256` | 0.8399 |
| `Assignment_3_v3.2-Arif-a1915248.ipynb` | `max_seq_length = 1024` | 0.8621 |
| `Assignment_3_v4.1-Arif-a1915248.ipynb` | Learning rate = 0.01 | 0.50 |
| `Assignment_3_v4.2-Arif-a1915248.ipynb` | Learning rate = 0.0001 | 0.8498 |
| `Assignment_3_v5.1-Arif-a1915248.ipynb` | No dropout (dropout = 0) | 0.8438 |
| `Assignment_3_v5.2-Arif-a1915248.ipynb` | Dropout = 0.2 | 0.8621 |

`Assignment_3_v2-Arif-a1915248.ipynb` needs the extra `transformers` package and downloads the `bert-base-uncased` tokenizer from the HuggingFace Hub on first run. All other variants use the same dependencies as `v1`.
