# Initial Project Audit

## 1. Executive Summary

This initial audit examines the existing codebase, artifacts, configuration, and research methodology in `/home/mml_07/Desktop/ATDL_Final_Project/JEPA_Model`.

The central research question is:
> **"Can JEPA-inspired self-supervised pretraining produce representations that improve downstream decision-making when fewer labeled examples are available?"**

The project structure established in the preliminary stage provides a working modular foundation. However, several critical methodological adjustments are required to ensure strict scientific validity, align the text JEPA formulation with established literature (data2vec and I-JEPA), evaluate against a proper Masked Language Model (MLM) baseline, and execute the label-efficiency fine-tuning experiments across decision types (`CHOICE`, `NOUL`, and `SCORE`).

---

## 2. Inventory of Existing Codebase and Components

### 2.1 Working Components
1. **PyTorch / CUDA Environment:**
   - Linux Ubuntu 6.8.0-52-generic, Intel Core i9-14900K CPU, NVIDIA RTX 4000 Ada Generation (20,475 MiB VRAM).
   - PyTorch 2.14.0+cu130 with native BF16 mixed-precision acceleration.
2. **Modular Architecture (`src/`):**
   - `encoder.py`: [`TextEncoder`](file:///home/mml_07/Desktop/ATDL_Final_Project/JEPA_Model/src/encoder.py) wrapping `distilbert-base-uncased` with mask-aware mean pooling and unit L2 normalization.
   - `predictor.py`: [`LatentPredictor`](file:///home/mml_07/Desktop/ATDL_Final_Project/JEPA_Model/src/predictor.py) MLP mapping context latents to target latents.
   - `jepa_model.py`: [`TextJEPA`](file:///home/mml_07/Desktop/ATDL_Final_Project/JEPA_Model/src/jepa_model.py) handling Context Encoder, Target Encoder (frozen, no optimizer gradients), and Exponential Moving Average ($\tau = 0.996$).
   - `losses.py`: [`JEPALoss`](file:///home/mml_07/Desktop/ATDL_Final_Project/JEPA_Model/src/losses.py) computing Normalized MSE.
   - `checkpointing.py`: Checkpoint saving and standalone reusable encoder export.
3. **Automated Test Suite (`tests/`):**
   - 17 unit tests passing in PyTest (padding invariance, gradient isolation, EMA correctness, split isolation).

---

## 3. What is Incomplete or Methodologically Questionable

### 3.1 Methodological Limitation in Context-Target Construction
- **Existing Approach:** Prefix/suffix split (first 60% of text as Context, trailing 40% as Target).
- **Critique:** While this guarantees non-identical text views, in natural language prose, a subsequent sentence often introduces completely new topic entities or facts not predictable from prior text.
- **Literature Alignment (data2vec / I-JEPA):**
  - In data2vec (Baevski et al., ICML 2022) and I-JEPA (Assran et al., CVPR 2023), the **Target Encoder receives the complete uncorrupted observation**, while the **Context Encoder receives a masked / corrupted view** of the same underlying observation.
  - The Predictor predicts the latent representation of the complete state from the masked representation.
  - This ensures that both views represent the exact same underlying state, eliminating artificial semantic divergence.

### 3.2 Sequence Length and Truncation
- Preliminary runs configured `max_seq_length = 128`.
- Token length analysis indicates that ~20% of states exceed 128 words.
- Increasing `max_seq_length` to 256 tokens captures over 90% of states without truncation while fitting comfortably within the 20 GB GPU VRAM.

### 3.3 Theoretical Calibration of Collapse Diagnostics
- The preliminary collapse monitor flagged `Mean per-dim std (0.028) < threshold (0.05)`.
- For an L2 unit-normalized vector in $\mathbb{R}^{768}$, the theoretical maximum standard deviation on a uniform sphere is $\sigma = \sqrt{1/768} \approx 0.0361$.
- A threshold of $0.05$ is physically impossible to achieve under unit normalization. The threshold is recalibrated to $0.015$ to detect genuine dimensional collapse.

### 3.4 Incomplete Experiments
- Full pretraining on all 116,655 unique training states (10 epochs) has not yet been executed.
- The standard **Masked Language Modeling (MLM)** pretraining baseline (Model C) on the exact same corpus has not been implemented.
- The downstream decision parsing, multi-task decision head, and label-efficiency protocol (10%, 25%, 50%, 100%) have not yet been built.

---

## 4. Planned File Modifications & Additions

| File | Action | Description |
| :--- | :--- | :--- |
| `src/view_generator.py` | **Modify** | Add span-masking corruption strategy (Context = masked state, Target = full state). |
| `configs/phase1.yaml` | **Modify** | Configure full-data training (10 epochs, seq len 256, checkpoints at 1, 3, 5, 8, 10). |
| `src/train_phase1.py` | **Modify** | Support span masking, full training budget, and checkpointing schedule. |
| `src/mlm_pretrain.py` | **Create** | Standard Masked Language Modeling pretraining baseline (Model C). |
| `src/downstream_dataset.py` | **Create** | Leakage-safe parser for `CHOICE`, `NOUL`, and `SCORE` decision tasks. |
| `src/downstream_model.py` | **Create** | Unified decision prediction head supporting multi-format evaluation. |
| `src/train_downstream.py` | **Create** | Label-efficiency fine-tuning runner across fractions (10%, 25%, 50%, 100%) and seeds. |
| `src/evaluate_downstream.py` | **Create** | Comprehensive evaluation on untouched test split (Accuracy, Macro-F1, MAE). |
| `tests/` | **Update** | Add unit tests for span masking, MLM pretraining, and downstream parser/evaluation. |

---

## 5. Estimated Compute Requirements

- **Workstation:** Intel Core i9-14900K, NVIDIA RTX 4000 Ada Generation (20 GB VRAM).
- **JEPA Pretraining (10 epochs on full corpus, ~106k pairs, batch size 32, seq len 256):**
  - ~3,300 steps/epoch @ ~40s/epoch = ~6.5 minutes.
  - VRAM: ~2.5 GB.
- **MLM Baseline Pretraining (10 epochs on same corpus):**
  - ~3,300 steps/epoch @ ~40s/epoch = ~6.5 minutes.
  - VRAM: ~2.5 GB.
- **Downstream Fine-Tuning:**
  - Fast convergence with early stopping (patience 2): ~1-2 minutes per run.
- **Total Compute Budget:** ~40-60 minutes total for all pretraining and label-efficiency experiments. All experiments are completely feasible within workstation resources.
