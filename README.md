
# Self-Pruning Neural Network – Case Study

## 1. Problem Overview

This project implements the **“Self-Pruning Neural Network”** case study specified in the Tredence AI Engineering Intern JD.  
The goal is to build a **feed-forward neural network** for **CIFAR-10 image classification** that can **prune its own connections during training**, instead of performing pruning as a separate post-processing step. [file:1]

The core idea:

- Each weight in the network has a **learnable gate** in the range **[0, 1]**.
- The effective weight is:  
  \[
  w_\text{effective} = w \times g
  \]
- If a gate **g → 0**, that connection is effectively **pruned**.
- During training, we optimize both:
  - **Classification loss** (cross-entropy),
  - **Sparsity loss** (L1 penalty on gate values),
  such that the network learns **which connections to keep and which to remove**. [file:1]

---

## 2. Architecture

### 2.1 High-level model

I use a simple **3-layer MLP** on top of flattened CIFAR-10 images:

- Input: **3 × 32 × 32 = 3072** features
- Hidden Layer 1: **128** units (prunable)
- Hidden Layer 2: **64** units (prunable)
- Output: **10** classes (prunable layer)

All fully connected layers are implemented as **`PrunableLinear`** instead of `nn.Linear`.

### 2.2 `PrunableLinear` layer

Each `PrunableLinear` has:

- `weight`: standard weight matrix of shape `[out_features, in_features]`
- `bias`: standard bias vector
- `gate_scores`: learnable parameter tensor with same shape as `weight`

**Forward pass:**

1. Convert scores to gates in `[0,1]`:
   ```python
   gates = torch.sigmoid(self.gate_scores)      # shape: [out_features, in_features]
   ```
2. Compute pruned weights:
   ```python
   pruned_weight = self.weight * gates          # element-wise
   ```
3. Apply linear transformation:
   ```python
   out = F.linear(x, pruned_weight, self.bias)
   ```

This ensures gradients flow through both **weights** and **gates**, and the optimizer can decide which connections to attenuate or prune.

### 2.3 Architecture diagram 
```mermaid
flowchart TD

A[Input Image (3x32x32)]
B[Flatten (3072)]

C[PrunableLinear (3072 -> 128)<br>weight_1<br>gate_scores_1<br>gates_1 = sigmoid(gate_scores_1)<br>effective_weight_1 = weight_1 * gates_1]

D[ReLU]

E[PrunableLinear (128 -> 64)<br>weight_2<br>gate_scores_2<br>gates_2 = sigmoid(gate_scores_2)<br>effective_weight_2 = weight_2 * gates_2]

F[ReLU]

G[PrunableLinear (64 -> 10)<br>weight_3<br>gate_scores_3<br>gates_3 = sigmoid(gate_scores_3)<br>effective_weight_3 = weight_3 * gates_3]

H[Logits (10 classes)]

A --> B --> C --> D --> E --> F --> G --> H
```
## 3. Loss Function and Sparsity

### 3.1 Classification + Sparsity loss

For each batch:

- **Classification loss**: standard cross-entropy:
  ```python
  cls_loss = CrossEntropyLoss(logits, labels)
  ```

- **Sparsity loss**: L1 penalty on all gate values:

  ```python
  sparse_loss = model.sparsity_loss()  # sum of all gate values across all layers
  sparse_loss = sparse_loss / num_gates  # normalize by number of gates
  ```

  Since `gates = sigmoid(gate_scores) ∈ [0,1]`, this is equivalent to an **L1 penalty** on the gates, which is known to encourage sparsity (push many values to exactly 0). [file:1]

- **Total loss**:
  ```python
  loss = cls_loss + lambda_value * sparse_loss
  ```

### 3.2 Why L1 on sigmoid gates encourages sparsity

- Each gate is constrained to be between **0 and 1**.
- We add the **sum of all gate values** to the loss.  
- Minimizing this sum makes the optimizer prefer gates **closer to 0**, unless keeping a gate non-zero significantly helps classification accuracy.
- L1 penalties are known to produce **sparse solutions**: many gates go to exactly 0, effectively pruning those connections. [file:1]

---

## 4. Training Strategy – Two-phase Schedule

### 4.1 Motivation

If we apply strong sparsity pressure (large λ) from the start:

- The network might **prune too aggressively** before it has learned good features,
- This can hurt both learning and final accuracy.

To avoid this, I use a simple **two-phase training schedule**:

1. **Phase 1 (warm-up)**:
   - Use a **small λ** (e.g. `1e-4`).
   - Goal: learn useful representations with minimal pruning.

2. **Phase 2 (pruning)**:
   - Increase λ to a **larger value** (e.g. `1e-3`, `1e-2`).
   - Goal: encourage many gates to drop towards 0, pruning less important connections.

This approach is very similar to how pruning is often done in practice:  
first train a decent model, then compress it.

### 4.2 Lambdas used

For the case study, I run training with three configurations:

```python
lambda_configs = [
    (1e-4, 1e-4),  # weak sparsity
    (1e-4, 1e-3),  # moderate sparsity
    (1e-4, 1e-2),  # strong sparsity
]
```

Where each tuple is `(lambda_phase1, lambda_phase2)`.

For each `(λ1, λ2)`:

- Phase 1 uses `λ1` for a few epochs,
- Phase 2 uses `λ2` for the remaining epochs.

---

## 5. How to Run – Local and Colab

### 5.1 Dependencies

Minimal dependencies (in `requirements.txt`):

```text
torch
torchvision
matplotlib
```

### 5.2 Run in Google Colab with GPU (recommended)

1. **Clone the repo:**

   ```python
   !git clone https://github.com/Prathiba2206/Self-pruning-NN.git
   %cd Self-pruning-NN
   ```

2. **Enable GPU:**

   - Runtime → Change runtime type → Hardware accelerator: **GPU** → Save.

3. **Install dependencies:**

   ```python
   !pip install -r requirements.txt
   ```

4. **Run training:**

   ```python
   !python train.py
   ```

This will:

- Download CIFAR-10,
- Train the self-pruning network with the two-phase λ schedule,
- Save results under `results/`:
  - `results.json` – list of dicts: `{lambda_phase1, lambda_phase2, test_accuracy, sparsity}`,
  - `gate_distribution.png` – histogram of final gate values for the best model.

### 5.3 Run locally (CPU or GPU)

If you have PyTorch installed locally:

```bash
pip install -r requirements.txt
python train.py
```

The device is chosen automatically:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 6. Outputs and Evaluation Metrics

The script generates:

1. **`results/results.json`**

   Contains a list like:

   ```json
   [
     {
       "lambda_phase1": 0.0001,
       "lambda_phase2": 0.0001,
       "test_accuracy": ...,
       "sparsity": ...
     },
     {
       "lambda_phase1": 0.0001,
       "lambda_phase2": 0.001,
       "test_accuracy": ...,
       "sparsity": ...
     },
     {
       "lambda_phase1": 0.0001,
       "lambda_phase2": 0.01,
       "test_accuracy": ...,
       "sparsity": ...
     }
   ]
   ```

   From this I construct the required table:

   | Lambda (Phase 2) | Test Accuracy (%) | Sparsity Level (%) |
   |------------------|-------------------|--------------------|
   | 1e-4             | ...               | ...                |
   | 1e-3             | ...               | ...                |
   | 1e-2             | ...               | ...                |

   - **Sparsity level** is defined as the **percentage of gates below a small threshold** (e.g., `1e-2`), exactly as described in the case study. [file:1]

2. **`results/gate_distribution.png`**

   - A histogram of all final gate values for the **best-performing model** (highest test accuracy).
   - In a successful self-pruning setup, the histogram shows:
     - A **large spike near 0** (pruned connections),
     - A cluster of values away from 0 (connections that stayed active).

---

## 7. How this relates to the JD case study structure

The JD/case study specifies: [file:1]

1. **Custom `PrunableLinear` layer**  
   - Implemented in `models.py` with `weight`, `bias`, and `gate_scores`.
   - Forward pass uses `sigmoid(gate_scores)` as gates and multiplies them with the weights.

2. **Sparsity regularization (L1 on gates)**  
   - `SelfPruningMLP.sparsity_loss()` computes the sum of all gate values across all prunable layers.
   - Total loss = classification loss + λ × normalized sparsity loss.

3. **Training on CIFAR-10**  
   - Dataset is loaded via `torchvision.datasets.CIFAR10`, as requested.

4. **Reporting**  
   - For **at least three λ values**, I report:
     - Test accuracy,
     - Sparsity level (% of gates below threshold),
     - And visualize gate value distribution with a histogram.

5. **Extension beyond the basic structure (two-phase schedule)**  
   - The case study describes the core mechanism; my implementation adds a simple but practical **two-phase training schedule**:
     - Phase 1 with small λ (feature learning),
     - Phase 2 with larger λ (aggressive pruning).
   - This makes the method closer to real-world pruning workflows and easier to explain:  
     “First learn, then prune”.

In summary, I followed the **specified structure** (custom prunable layer, L1 gating, CIFAR-10, evaluation across λ), and added a **two-phase λ schedule** as a small, practical enhancement that improves stability and clarity of the sparsity–accuracy trade-off.

---

## 8. Files in this repository

- `models.py`  
  - Implements `PrunableLinear` and `SelfPruningMLP`.

- `utils.py`  
  - Helper functions for accuracy and sparsity computation.

- `train.py`  
  - CIFAR-10 dataloaders,
  - Two-phase training loop for multiple λ configurations,
  - Result logging and gate distribution plotting.

- `requirements.txt`  
  - Minimal Python dependencies.

- `results/` (generated after training)  
  - `results.json` – numerical results for λ vs accuracy vs sparsity.
  - `gate_distribution.png` – gate value histogram for best model.

- `README.md`  
  - This document: problem explanation, architecture, training details, and relation to the JD case study.
```

***

If you want, after you run training once in Colab and get your actual numbers, you can:

- Fill in the table under **Section 6** with real `Test Accuracy` and `Sparsity` values from `results.json`.
- Add a short 2–3 line “Results & Observations” section, and I can help you phrase it in a very interview‑friendly way.
