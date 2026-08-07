# MNIST CNN From Scratch (NumPy) — Assignment_2_v3-Arif-a1915248

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification **entirely from scratch using NumPy** — forward pass, backpropagation, and the SGD weight update are all hand-written (no `torch.nn`, no autograd). PyTorch/torchvision are used only to download MNIST and serve it in mini-batches via `DataLoader`.

This README covers `Assignment_2_v3-Arif-a1915248.ipynb` only.

## Summary

`Assignment_2_v3-Arif-a1915248.ipynb` builds a 2 conv + 2 pool + 1 FC CNN and trains it on MNIST for 5 epochs using manually-derived gradients and mini-batch SGD (batch size 32) with He (Kaiming) weight initialisation. It trains on the raw, un-augmented MNIST images.

## Project Files

- `Assignment_2_v3-Arif-a1915248.ipynb` — the notebook (dataset loading, model definition, training/evaluation calls).
- `train_utils.py` — **required sibling module**. Defines the shared `one_hot()`, `train()`, and `test()` functions that the notebook imports (`from train_utils import train, test`); the model class itself (`ManualCNN`) still lives inline in the notebook.

## Requirements

- Python 3.9+
- A Google Colab runtime with a linked Google Drive (see "How to Run" below)
- Packages (pre-installed on Colab, or install manually):
  - `numpy`
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `scikit-learn`

## Dataset

- **Dataset**: MNIST handwritten digits (10 classes, digits 0–9), loaded via `torchvision.datasets.MNIST` with `download=True`. The dataset is fetched automatically into a local `./data` folder on first run.
- **Split**: uses the standard MNIST split provided by torchvision — **60,000 training images** and **10,000 test images**. No separate held-out validation set is carved out of the training data; per-epoch "validation" metrics reported during training are computed by re-running inference over the full training set, not a distinct validation split.
- **Preprocessing**: `Resize(28)` → `ToTensor()` (scales pixels to `[0, 1]`, shape `1×28×28`). No data augmentation is applied.

## Step-by-step: How to Run

This notebook is written for **Google Colab**: its first cell mounts Google Drive and adds a Drive folder to `sys.path` so it can `import train_utils`, and it saves pickled results to a Drive path (`RESULT_PATH`) rather than a local folder.

1. Upload `Assignment_2_v3-Arif-a1915248.ipynb` and `train_utils.py` to the same Google Drive folder, e.g. `MyDrive/Colab Notebooks/APPLIED AIML 2/Assignment 2` (this is the path hard-coded in the notebook's `sys.path.insert(...)` call and `RESULT_PATH`; edit those two lines if you use a different folder).
2. Open the notebook in Google Colab (double-click it in Drive, or upload it at colab.research.google.com).
3. Run all cells top to bottom:
   - The first cell prompts a Google Drive authorization popup — approve it so the notebook can mount `/content/drive` and import `train_utils`.
   - Downloads MNIST (first run only), builds the ManualCNN, trains for 5 epochs, evaluates on the test set, and saves pickled metrics to `RESULT_PATH` on Drive.

Note: training loops over every image individually (batch accumulation of per-sample gradients), so a full run takes roughly 20–25 minutes.

## Model Architecture

`Assignment_2_v3-Arif-a1915248.ipynb` uses a `ManualCNN` class:

| Layer | Details |
|---|---|
| Input | 1×28×28 grayscale image |
| Conv Layer 1 | 6 filters, 3×3 kernel, He (Kaiming) init, stride 1, no padding → ReLU |
| Max Pool 1 | 2×2 window |
| Conv Layer 2 | 16 filters, 3×3 kernel, He (Kaiming) init → ReLU |
| Max Pool 2 | 2×2 window |
| Flatten | → 400 features |
| Fully Connected | 400 → 128, He init → ReLU |
| Output | 128 → 10, Softmax |

Implementation notes:
- Convolution is implemented as cross-correlation using `numpy.lib.stride_tricks.sliding_window_view` combined with `einsum`, rather than explicit nested loops.
- Max pooling stores an argmax mask so the pooling gradient can be routed back to the correct input position during backprop.
- Loss is manual categorical cross-entropy against one-hot encoded labels.
- Backpropagation is fully manual: gradients are derived layer-by-layer and returned as a dictionary.
- Weight updates use mini-batch SGD: per-sample gradients are accumulated over a batch, averaged, and applied once per batch

## Hyperparameters

### `Assignment_2_v3-Arif-a1915248.ipynb`
| Hyperparameter | Value |
|---|---|
| Learning rate | 0.01 |
| Batch size | 32 |
| Epochs | 5 |
| Hidden layer size | 128 |
| Pooling window | 2×2 |
| Weight init | He (Kaiming) |
| Data augmentation | None |

## Results

### `Assignment_2_v3-Arif-a1915248.ipynb`
- Final training loss (epoch 5/5): **0.10**
- Test set — weighted avg precision/recall/f1: **~0.97**, overall **accuracy: 97.34%**
- Training time: ≈1266 s (≈21.1 min)
- Testing time: ≈13.0 s

## Further Variants

`Assignment_2_v4-Arif-a1915248.ipynb` through `Assignment_2_v6-Arif-a1915248.ipynb` (data augmentation, weight decay, and dropout variants respectively) are not covered in this README — see the report `Assignment_2-Arif-a1915248.pdf` for details.
