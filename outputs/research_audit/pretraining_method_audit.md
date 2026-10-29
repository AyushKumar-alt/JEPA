# Pretraining Method Audit & Theoretical Foundation

## 1. Grounding in Recent Literature: data2vec and I-JEPA

The implementation of the lightweight text JEPA model is grounded in the foundational principles established by two key self-supervised learning milestones:

1. **Baevski et al. (ICML 2022):** *"data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language."*
   - **Key Principle:** Rather than predicting discrete modality-specific tokens (e.g., words via cross-entropy in masked language modeling), data2vec predicts **continuous latent representations** produced by a self-teacher (target encoder) from a masked input representation (context encoder).
   - **Target Encoder Update:** The target encoder is parameterized as an Exponential Moving Average (EMA) of the online context encoder weights:
     $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{target}} + (1 - \tau) \theta_{\text{context}}$$
   - **View Construction:** The target encoder processes the complete, unmasked sequence, while the context encoder processes a sequence corrupted by span masking.

2. **Assran et al. (CVPR 2023):** *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)."*
   - **Key Principle:** A predictor network maps context representations into the target representation space. Gradients flow exclusively through the context encoder and predictor; the target encoder is entirely frozen with respect to optimizer descent.
   - **Elimination of Trivial Collapse:** The architecture avoids trivial representation collapse through the asymmetric predictor, target gradient stopping, EMA parameter momentum, and normalized latent regression.

---

## 2. Text JEPA View Construction: Transition to Span-Masking

### 2.1 Limitation of Prior Heuristic View Splitting
The preliminary implementation utilized a sentence-prefix/suffix split (first 60% of sentences as Context, trailing 40% as Target). While satisfying the formal requirement of non-identical views ($x_{\text{context}} \neq x_{\text{target}}$), in diverse text datasets (e.g., bug reports, reasoning puzzles, dialogues), later sentences frequently introduce new entities, external consequences, or topics that are fundamentally uninferrable from the prefix alone.

### 2.2 Refined Span-Masking Design
In accordance with data2vec:
- **Target View ($x_{\text{target}}$):** The complete, uncorrupted state text $S$.
- **Context View ($x_{\text{context}}$):** The same state text with contiguous token spans replaced by a dedicated `[MASK]` token (masking ratio: 25%, maximum span length: 4 words).
- **Semantic Guarantee:** Both views represent the exact same underlying state $S$. The objective teaches the encoder to reconstruct the global latent semantics of the state from partial, masked observations.
- **Short-State Handling:** States with $< 8$ words (~15% of the dataset) are excluded because masking a single word alters a disproportionate fraction of the total semantic payload.

---

## 3. Sequence Length Distribution and Truncation Analysis

An empirical token length analysis was performed using the DistilBERT tokenizer across a sample of 5,000 states from `data/bev-decision/train.parquet`:

| Metric | Measured Value |
| :--- | :---: |
| **Minimum Tokens** | 4 |
| **25th Percentile** | 88 |
| **Median** | 153 |
| **Mean** | 226.6 |
| **75th Percentile** | 319 |
| **90th Percentile** | 485 |
| **95th Percentile** | 629 |
| **99th Percentile** | 1,009 |
| **Maximum Tokens** | 2,160 |

### Truncation Rates Across Candidate Sequence Limits:
- **At 128 Tokens:** 56.66% truncated (more than half the dataset truncated).
- **At 256 Tokens:** 35.62% truncated (retains complete text for ~65% of states).
- **At 384 Tokens:** 17.02% truncated.
- **At 512 Tokens:** 8.50% truncated (architectural limit for DistilBERT).

**Decision:** The sequence length is increased from 128 to **256 tokens** for full-data pretraining, striking an optimal trade-off between semantic retention, memory footprint, and training throughput. Dynamic batch padding is used to eliminate wasted computation on shorter sequences.

---

## 4. Architecture and Mathematical Mechanics

```
Context Text: "[MASK] patient reported [MASK] headaches..."
        |
        v
 [ Context Encoder (DistilBERT + Masked Mean Pooling + LayerNorm + L2) ]
        |
        v
  z_context (768-D, unit norm)  ----->  [ Predictor MLP (2 Layers + Skip) ]
                                                       |
                                                       v
                                            z_pred (768-D, unit norm)
                                                       |
                                                       v  <--- Latent Prediction Loss
Target Text: "The patient reported experiencing..."   |       (Normalized MSE)
        |                                              |
        v                                              |
 [ Target Encoder (Frozen Clone, EMA tau=0.996) ]      |
        |                                              |
        v                                              |
  z_target (768-D, unit norm, no grad)  ---------------+
```

### 4.1 Mask-Aware Mean Pooling Padding Invariance
Token embeddings $H \in \mathbb{R}^{B \times L \times D}$ and attention masks $M \in \{0, 1\}^{B \times L}$:
$$\bar{h} = \frac{\sum_{i=1}^L h_i \cdot M_i}{\sum_{i=1}^L M_i}$$
Unit test `test_masked_mean_pooling_padding_invariance` validates that appending arbitrary pad tokens with zero mask produces identical pooled representations up to machine precision ($\Delta < 10^{-6}$).

### 4.2 Latent Prediction Loss
$$\mathcal{L}(z_{\text{pred}}, z_{\text{target}}) = \frac{1}{D} \sum_{d=1}^D \left( \frac{z_{\text{pred}, d}}{\|z_{\text{pred}}\|_2} - \frac{z_{\text{target}, d}}{\|z_{\text{target}}\|_2} \right)^2$$
Target latents are detached (`torch.no_grad()`), preventing target encoder gradients.

### 4.3 Geometrical Calibration of Collapse Indicators
On a 768-dimensional unit hypersphere:
- Expected standard deviation per dimension for a uniform spherical distribution:
  $$\sigma_{\text{theoretical}} = \sqrt{\frac{1}{768}} \approx 0.0361$$
- **Recalibrated Thresholds:**
  - Alert if $\text{mean}(\sigma_d) < 0.012$ (indicating collapse to an excessively narrow cluster).
  - Alert if $> 30\%$ of dimensions have variance $< 0.001$.
  - Alert if mean pairwise cosine similarity $S_{\text{pairwise}} > 0.90$.
