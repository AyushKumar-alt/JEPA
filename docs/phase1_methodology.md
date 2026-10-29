# Phase 1 Methodology: JEPA-Inspired Self-Supervised Text Representation Learning

## 1. Research Motivation & Objective

Self-supervised representation learning in computer vision has seen major advances through Joint-Embedding Predictive Architectures (JEPA) such as I-JEPA and V-JEPA (LeCun, 2022; Assran et al., 2023). Unlike generative models that predict every masked pixel or token in input space, JEPA learns strictly in the abstract **latent representation space**: given a context view of a state, the model is trained to predict the latent representation of a held-out target view derived from the same state.

The overarching course project for *Advanced Topics in Deep Learning* (ATDL) investigates:
> *"Can JEPA-inspired self-supervised text representation learning produce useful latent representations from unlabeled state text, potentially improving downstream structured decision-making when labeled examples are scarce?"*

### Scope of Phase 1
Phase 1 focuses **strictly** on the self-supervised representation learning stage:
- **No decision labels or questions are used during training or model selection.**
- The model trains exclusively on state-derived text from the BEV Decision dataset.
- Phase 1 does **not** evaluate decision accuracy or claim that representations are superior for downstream tasks. It establishes the representation learning pipeline, verifies optimization stability, monitors representation collapse, and persists a reusable context encoder for subsequent Phase 2 evaluation.

---

## 2. Dataset Structure and Strict Leakage Prevention

### 2.1 Provenance and Dataset Files
The dataset resides in `data/bev-decision/`:
- `train.parquet`: 125,614 rows (50.7 MB).
- `test.parquet`: 24,386 rows (5.7 MB).

### 2.2 Schema and Field Audit
Each split contains five columns:
1. `state` (`large_string`, 0 nulls): Natural language text describing environment states, scenarios, customer requests, bug reports, reasoning steps, or dialogues.
2. `questions_json` (`large_string`, 0 nulls): JSON objects containing decision questions, instructions, criteria, choices, and ground-truth decision labels.
3. `domain` (`string`, 0 nulls): Categorical domain identifier (e.g., Sentiment, Spatial and logical reasoning, Retail, Support, Scientific, Software engineering).
4. `question_types` (`list<string>`, 0 nulls): Task types (e.g. `['choice']`).
5. `question_count` (`int16`, 0 nulls): Count of questions per row.

### 2.3 Strict Data Leakage Safeguards
To guarantee scientific integrity:
- **Only the `state` field is used.**
- All fields containing decision labels (`questions_json`, `question_types`, `question_count`, `choice`, `score`, `noul`) are completely excluded at the parsing boundary via `src.data_processing.load_and_clean_states`.
- Unit tests (`tests/test_data_processing.py`) audit pretraining records to guarantee zero label contamination.
- State text audit confirmed that decision prompts and instructions are not embedded in `state`.
- **The original `test.parquet` split remains completely untouched.** It is never passed into pretraining, validation, or checkpoint selection.

### 2.4 Partitioning and Deduplication Strategy
In the training split, multiple rows share identical `state` text because multiple decision questions were associated with the same scenario:
- Unique states in training: 116,671 out of 125,614 rows (92.88%).
- **Grouping Policy:** Rows are grouped by their unique state hash (`state_hash`) before splitting.
- **Internal Split:** 90% internal training, 10% internal validation using a fixed random seed (42).
- **Leakage Invariance:** Exact duplicate states are constrained to either train or validation, ensuring zero state overlap ($Train \cap Val = \emptyset$).

---

## 3. Context-Target View Construction

JEPA requires deriving two non-identical, informative views from the same underlying state: a **Context View** $x_{\text{context}}$ and a **Target View** $x_{\text{target}}$.

### 3.1 Partitioning Strategy
Implemented in `src.view_generator.ViewGenerator`:
1. **Multi-Sentence Splitting:**
   - When states contain $\ge 2$ sentences (e.g., customer complaints, reasoning puzzles, code descriptions):
   - Sentences are split along punctuation boundaries (`[.!?\n]`).
   - The prefix $k = \lceil N \times 0.6 \rceil$ sentences form the Context View.
   - The remaining $N - k$ sentences form the Target View.
2. **Single-Sentence Word-Span Splitting:**
   - For single-sentence states with sufficient length ($\ge 8$ words):
   - The first 60% of words form the Context View; the remaining 40% form the Target View.
3. **Short-State Filtering Policy:**
   - States with $< 8$ words cannot form non-identical, non-trivial context and target views without trivial memorization.
   - Such states (~15% of rows, often short phrases like `"Things that matter."` or `"twist open the ouzo!"`) are systematically excluded and logged.
4. **Data-Quality Guarantees:**
   - Both Context and Target must be non-empty.
   - Context and Target must not be identical ($Context \neq Target$).
   - Context and Target have non-overlapping spans.

### 3.2 Rationale for JEPA Objective
Predicting the latent representation of subsequent state dynamics from preceding context forces the encoder to capture high-level semantic intent, causal transitions, and state constraints, rather than superficial lexical matching.

---

## 4. Architecture Specifications

The lightweight JEPA model is implemented in `src.jepa_model.TextJEPA`:

```
+-------------------------------------------------------------+
|                        TextJEPA                             |
|                                                             |
|   Context Text ---> [ Context Encoder ] ---> z_context (768)|
|                            |                       |        |
|                   (AdamW Gradients)                v        |
|                            |               [   Predictor   ]|
|                            v                       |        |
|                     theta_context                  v        |
|                            |                  z_pred (768)  |
|                         (EMA tau=0.996)            |        |
|                            v                       v        |
|   Target Text  ---> [ Target Encoder  ] ---> z_target (768) |
|                     (Frozen, No Grad)              |        |
|                                                    v        |
|                                           Normalized MSE    |
+-------------------------------------------------------------+
```

### 4.1 Context Encoder (`src.encoder.TextEncoder`)
- **Backbone:** Compact pretrained Transformer (`distilbert-base-uncased`, 66M parameters, hidden size 768).
- **Pooling:** Mask-aware mean pooling:
  $$\bar{h} = \frac{\sum_{i=1}^L h_i \cdot m_i}{\sum_{i=1}^L m_i}$$
  Strictly ignores padding tokens so pooled latents are invariant to batch padding length (verified in unit tests).
- **Projection & Normalization:** `LayerNorm(768)` followed by optional L2 unit normalization.

### 4.2 Target Encoder
- Exact architectural clone of the Context Encoder.
- Initialized from the Context Encoder parameters.
- **Strictly frozen:** `requires_grad = False` across all parameters.
- Target branch is evaluated inside `torch.no_grad()`.
- Updated exclusively via Exponential Moving Average (EMA):
  $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{target}} + (1 - \tau) \theta_{\text{context}}$$
  where $\tau = 0.996$.

### 4.3 Predictor (`src.predictor.LatentPredictor`)
- Lightweight MLP mapping context latent space to target latent space:
  $$\text{Linear}(768, 768) \to \text{LayerNorm}(768) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(768, 768) \to \text{LayerNorm}(768)$$
- Includes a residual connection $z_{\text{context}} + \text{MLP}(z_{\text{context}})$.
- Output is L2-normalized matching target latent space.

---

## 5. Training Objective & Representation Collapse Diagnostics

### 5.1 Latent Prediction Loss (`src.losses.JEPALoss`)
The model optimizes Normalized Mean Squared Error between the predicted target representation $\hat{z}$ and the target encoder representation $z_{\text{target}}$:
$$\hat{u} = \frac{\hat{z}}{\|\hat{z}\|_2}, \quad u_{\text{target}} = \frac{z_{\text{target}}}{\|z_{\text{target}}\|_2}$$
$$\mathcal{L}(\hat{u}, u_{\text{target}}) = \frac{1}{D} \sum_{d=1}^D (\hat{u}_d - u_{\text{target}, d})^2$$
Normalized MSE provides scale invariance, prevents norm shrinkage, and bounds the loss in $[0, 4]$.

### 5.2 Representation Collapse Monitoring
To prevent or detect trivial constant representations ($z \to c$ across all inputs):
1. **Per-Dimension Standard Deviation:** $\sigma_d = \text{std}_{B}(z_{:, d})$. Alert triggered if $\text{mean}(\sigma_d) < 0.05$.
2. **Collapsed Dimension Fraction:** Fraction of dimensions where $\sigma_d < 0.01$. Alert triggered if $> 30\%$.
3. **Mean Pairwise Cosine Similarity:**
   $$S_{\text{pairwise}} = \frac{1}{B(B-1)} \sum_{i \neq j} \frac{z_i \cdot z_j}{\|z_i\|_2 \|z_j\|_2}$$
   Alert triggered if $S_{\text{pairwise}} > 0.95$ (indicating inputs map to nearly identical vectors).
4. **Mean Latent Vector Norm:** Monitored to confirm unit norm stability.

---

## 6. Optimization & Checkpointing

- **Optimizer:** AdamW with decoupled parameter groups:
  - Context Encoder: $\text{lr} = 3 \times 10^{-5}$
  - Predictor: $\text{lr} = 1 \times 10^{-4}$
  - Weight decay: $0.01$
- **Scheduler:** Cosine annealing with linear warmup.
- **Precision:** Native BF16 mixed precision (`torch.bfloat16`) leveraging the RTX 4000 Ada Tensor Cores.
- **Gradient Clipping:** Maximum grad norm $1.0$.
- **Validation:** Computed at the end of each epoch without gradients or EMA updates.
- **Persistence:**
  - `checkpoint_latest.pt`: Full state (model, optimizer, scheduler, epoch, step).
  - `checkpoint_best.pt`: Best validation checkpoint.
  - `reusable_encoder/`: Standalone directory containing only the trained Context Encoder weights (`context_encoder.pt`), configuration (`encoder_config.json`), and tokenizer files for direct reuse in Phase 2.

---

## 7. Hardware & Software Environment

- **GPU:** NVIDIA RTX 4000 Ada Generation (20,475 MiB VRAM).
- **CPU:** Intel Core i9-14900K (24 cores / 32 threads).
- **RAM:** 31 GB.
- **OS:** Ubuntu Linux (Kernel 6.8.0-52-generic).
- **PyTorch:** 2.14.0+cu130 with CUDA 13.2.
- **Hugging Face Transformers:** 5.17.0.

---

## 8. Known Limitations & Phase 2 Roadmap

### Limitations
1. Text JEPA views rely on syntactic sentence boundaries and word-span splits, which may occasionally split within tightly coupled idiomatic clauses.
2. Extremely short states ($<8$ words) are discarded from self-supervised training.
3. Pretraining uses DistilBERT backbone weights; therefore, Phase 1 represents self-supervised continual domain adaptation rather than training from scratch.

### Downstream Integration (Phase 2)
In Phase 2, the reusable context encoder (`outputs/phase1/checkpoints/reusable_encoder`) will be loaded with frozen and unfrozen heads to evaluate linear probe accuracy and sample-efficient fine-tuning on decision tasks across low-label regimes ($1\%, 5\%, 10\%, 100\%$).
