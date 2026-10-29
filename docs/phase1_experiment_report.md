# Phase 1 Experiment Report: Self-Supervised JEPA Pretraining

## 1. Executive Summary

This report documents the implementation, unit testing, pilot verification, full self-supervised training, and diagnostic audit for **Phase 1** of the academic project:
> *"Design and Development of a Lightweight JEPA-Inspired Model for Structured Decision-Making"*

Phase 1 successfully developed and verified a label-free Joint-Embedding Predictive Architecture (JEPA) text representation learning pipeline trained strictly on state text from the BEV Decision dataset. All mandatory data-leakage safeguards, gradient isolation rules, Exponential Moving Average (EMA) parameter updates, and collapse diagnostics were verified through automated tests and experimental execution.

---

## 2. Experimental Setup

### 2.1 Hardware and Runtime Environment
- **Workstation:** Ubuntu Linux 6.8.0-52-generic, Intel Core i9-14900K (24 cores / 32 threads, 31 GB RAM).
- **GPU Accelerator:** NVIDIA RTX 4000 Ada Generation (20,475 MiB VRAM).
- **Software Stack:** Python 3.12.3, PyTorch 2.14.0+cu130 (CUDA 13.2 capability, cuDNN 9.2), Hugging Face Transformers 5.17.0, PyArrow 26.0.0, PyTest 9.2.0, Matplotlib 3.11.2.

### 2.2 Dataset Schema & Split Accounting
- **Data Source:** `data/bev-decision/` (`train.parquet`: 125,614 rows; `test.parquet`: 24,386 rows).
- **Data Leakage Safeguards:**
  - `questions_json`, decision questions, question types, and decision labels were completely excluded from pretraining.
  - Zero question instruction leakage was detected in the state text across sample audits.
  - The original `test.parquet` split remained completely untouched.
- **Internal Partitioning:**
  - States were deduplicated and grouped by unique hash (`state_hash`) before splitting.
  - Split ratio: 90% internal train, 10% internal validation (seed 42).
  - Evaluated on a reproducible training subset of 20,000 states: 16,841 training state groups and 1,871 validation state groups.
  - 16,701 valid training context-target pairs were generated (140 rejected due to insufficient length $<8$ words).
  - 1,847 validation pairs were generated (24 rejected).

---

## 3. Unit Test Suite Results

The comprehensive test suite in `tests/` was executed using PyTest:
```bash
.venv/bin/pytest tests/ -v
```
**Result: 17 passed in 9.79 seconds (100% pass rate).**

| Test Module | Test Case | Target Property Verified | Result |
| :--- | :--- | :--- | :--- |
| `test_data_processing.py` | `test_parquet_files_readable` | Parquet integrity and schema presence | **PASSED** |
| `test_data_processing.py` | `test_state_extraction_drops_labels` | Complete absence of `questions_json` and labels | **PASSED** |
| `test_data_processing.py` | `test_train_val_split_no_overlap` | Zero state hash or text overlap between train and val | **PASSED** |
| `test_data_processing.py` | `test_original_test_split_remains_untouched` | Test split row count intact (24,386 rows untouched) | **PASSED** |
| `test_view_generator.py` | `test_view_generation_valid_pair` | Context $\neq$ Target, non-empty, minimum length | **PASSED** |
| `test_view_generator.py` | `test_view_generation_short_state_handling` | Graceful exclusion of states $<8$ words | **PASSED** |
| `test_view_generator.py` | `test_view_generation_deterministic` | View generation invariance under fixed seed | **PASSED** |
| `test_view_generator.py` | `test_view_generation_batch_stats` | Batch diagnostic accounting | **PASSED** |
| `test_encoder.py` | `test_masked_mean_pooling_padding_invariance` | Pooled vectors invariant to sequence padding length | **PASSED** |
| `test_encoder.py` | `test_text_encoder_output_dimensions` | Latent shape (B, 768) and unit L2 normalization | **PASSED** |
| `test_jepa_model.py` | `test_jepa_forward_shapes` | Context, Target, and Predictor output dimensions match | **PASSED** |
| `test_jepa_model.py` | `test_gradient_isolation` | Gradients reach context & predictor; zero grads to target | **PASSED** |
| `test_jepa_model.py` | `test_ema_mathematical_correctness` | Exact numerical verification of $\theta_t \leftarrow \tau \theta_t + (1-\tau)\theta_c$ | **PASSED** |
| `test_training.py` | `test_mini_training_step` | Finite loss backward pass and optimizer step | **PASSED** |
| `test_training.py` | `test_validation_does_not_alter_parameters` | Weights unchanged during validation forward passes | **PASSED** |
| `test_checkpointing.py` | `test_checkpoint_save_and_reload` | Bitwise parameter equality on checkpoint reload | **PASSED** |
| `test_checkpointing.py` | `test_reusable_encoder_export_and_inference` | Standalone context encoder persistence & inference | **PASSED** |

---

## 4. Pilot Training Results

Before launching the full pretraining experiment, a pilot was executed on 256 states for 13 optimization steps:
- **Duration:** 1.69 seconds.
- **Initial Loss:** 0.0031 $\to$ **Final Loss:** 0.0009 (finite, monotonic descent).
- **Target Encoder Gradient Check:** Exactly 0 parameters received optimizer gradients.
- **Context & Predictor Gradient Check:** Non-zero gradients verified across all learnable layers.
- **EMA Parameter Delta:** Cumulative absolute parameter delta $\Delta = 2.9794 \times 10^{0}$, confirming target parameters update dynamically via context EMA.
- **Peak VRAM:** 1,803.2 MB (9% of GPU capacity).
- **Pilot Checkpoint:** Persisted to `outputs/phase1/checkpoints/checkpoint_pilot.pt`.

---

## 5. Full Phase 1 Pretraining Results

The full Phase 1 experiment was trained over 3 complete epochs using native BF16 mixed precision and AdamW:
- **Total Parameters:** 133,913,088
- **Trainable Parameters:** 67,548,672 (Context Encoder: 66,364,416; Predictor: 1,184,256).
- **Frozen Target Parameters:** 66,364,416.
- **Batch Size:** 32 (522 batches per epoch, 1,566 total optimization steps).
- **Total Training Duration:** 104.6 seconds (~34.8 seconds per epoch).
- **Peak GPU VRAM:** 2,010.3 MB.

### Epoch Metrics Progression

| Epoch | Train Loss (Norm MSE) | Val Loss (Norm MSE) | Mean Latent Norm | Mean Per-Dim Std | Mean Pairwise Sim | Duration (s) | Best Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **0.000223** | **0.000114** | **1.000** | **0.0277** | **0.328** | 33.3s | **Saved (Best)** |
| 2 | 0.000153 | 0.000164 | 1.000 | 0.0301 | 0.225 | 33.3s | No |
| 3 | 0.000194 | 0.000209 | 1.000 | 0.0305 | 0.210 | 33.5s | No |

---

## 6. Analysis of Representation Diversity & Collapse Diagnostics

### Did Representation Collapse Occur?
**No.** The empirical metrics demonstrate that the learned representations did not collapse:
1. **Pairwise Cosine Similarity:**
   - Early training steps exhibited higher similarity (~0.75).
   - As training progressed, mean pairwise cosine similarity dropped steadily:
     $$\text{Step 50: } 0.751 \longrightarrow \text{Epoch 1: } 0.328 \longrightarrow \text{Epoch 2: } 0.225 \longrightarrow \text{Epoch 3: } 0.210$$
   - A mean pairwise similarity of $\sim 0.21$ in 768-dimensional space indicates high angular dispersion across diverse state texts, far below the collapse warning threshold ($>0.95$).
2. **Per-Dimension Standard Deviation & Unit-Sphere Geometry:**
   - In 768-dimensional space with L2 unit normalization ($\|z\|_2 = 1$), a perfectly uniform distribution across the sphere has theoretical per-dimension standard deviation:
     $$\sigma_{\text{theoretical}} = \sqrt{\frac{1}{D}} = \sqrt{\frac{1}{768}} \approx 0.0361$$
   - The observed per-dimension standard deviation rose from $0.0162$ at step 50 to $0.0305$ at epoch 3, approaching the geometric upper bound.
   - The initial heuristic threshold of $0.05$ (calibrated for unnormalized representations) triggered warning logs, but mathematical inspection confirms that $0.0305$ is near-optimal for unit-normalized vectors.
3. **Latent Norm Stability:**
   - Mean latent vector norm remained exactly $1.000$ throughout, with zero norm explosion or vanishing.

---

## 7. Artifacts and Output Files

All experiment artifacts have been saved under `outputs/phase1/`:

### 7.1 Checkpoints (`outputs/phase1/checkpoints/`)
- `checkpoint_pilot.pt` (1.07 GB): Initial pilot checkpoint.
- `checkpoint_best.pt` (1.07 GB): Best validation checkpoint (Epoch 1, Val Loss 0.000114).
- `checkpoint_latest.pt` (1.07 GB): Final training checkpoint (Epoch 3).
- **`reusable_encoder/`** (265 MB): Independent, turnkey representation encoder directory containing:
  - `context_encoder.pt`: Trained context encoder weights.
  - `encoder_config.json`: Model architecture, latent dim, pooling type, and tokenizer config.
  - `tokenizer_config.json`, `tokenizer.json`: Tokenizer files for immediate Phase 2 use.

### 7.2 Metrics (`outputs/phase1/metrics/`)
- `dataset_inspection.json`: Complete profile of splits, distributions, and leakage audit.
- `phase1_metrics.json`: Epoch-by-epoch loss and collapse telemetry.
- `phase1_metrics.csv`: Tabular metric logs.
- `evaluation_summary.json`: Verification and test audit of the reusable encoder.

### 7.3 Plots (`outputs/phase1/plots/`)
- `loss_vs_epoch.png`: Training and validation loss trajectory.
- `latent_std_vs_epoch.png`: Latent per-dimension variance progression.
- `latent_norm_vs_epoch.png`: Unit norm verification over epochs.
- `pairwise_similarity_vs_epoch.png`: Cosine dispersion demonstrating lack of collapse.

---

## 8. Limitations & Recommended Next Steps

### Limitations
1. Pretraining utilized DistilBERT initialization; therefore, the representations represent self-supervised adaptation of an existing language model rather than tabula rasa learning.
2. Short states ($<8$ words) were filtered out to prevent trivial context-target identity.

### Transition to Phase 2
The reusable encoder is ready for downstream evaluation. In Phase 2:
1. Attach linear probes or lightweight MLP heads for decision prediction.
2. Compare Phase 1 JEPA representations against baseline encoders (untrained DistilBERT, supervised baseline).
3. Evaluate label efficiency under restricted labeled data regimes (e.g., 1%, 5%, 10%, 25%, 100% of labels).
