# Downstream Decision Dataset Audit & Formulation Report

## 1. Executive Summary

This report establishes the schema, leakage safeguards, formatting, and subset sampling methodology for Phase 2 structured decision-making on the BEV Decision Mix dataset.

The downstream objective evaluates whether representations learned via self-supervised pretraining (JEPA vs. Masked Language Modeling vs. Supervised Baseline) improve sample-efficient decision accuracy across restricted label regimes ($10\%, 25\%, 50\%, 100\%$).

---

## 2. Decision Task Types and Real Schema Inspection

The official dataset contains 239,747 individual decision questions across 125,614 training state rows. Inspection of `questions_json` confirms three distinct decision paradigms:

### 2.1 `CHOICE` (100,870 instances, 42.1%)
- **Definition:** Bounded multiple-choice decision selecting among 2 to 8 candidate options.
- **Criteria Structure:** Key-value mapping of candidate identifiers to detailed semantic criteria (e.g., `{"background": "Background or prior work", "method": "Method or procedure", "result": "Result or finding"}`).
- **Ground Truth Target:** Target key string (e.g., `"background"`), mapped to an option index $y \in \{0, \dots, K-1\}$.
- **Formulation:** Multi-class classification via Cross-Entropy loss.

### 2.2 `NOUL` (93,918 instances, 39.2%)
- **Definition:** Binary proposition verification ("Yes" vs. "No" / True vs. False).
- **Target:** Boolean value (`True` or `False`), mapped to $y \in \{0, 1\}$.
- **Formulation:** Binary classification evaluated via Cross-Entropy, Accuracy, and Macro-F1.

### 2.3 `SCORE` (44,959 instances, 18.7%)
- **Definition:** Ordered evaluation along a discrete 5-point scale (levels 0 to 4).
- **Criteria Structure:** Ordered list of qualitative rating descriptions (`["Very negative", "Negative", "Neutral", "Positive", "Very positive"]`).
- **Ground Truth Target:** Integer level $y \in \{0, 1, 2, 3, 4\}$.
- **Formulation & Metrics:** Evaluated via Classification Accuracy and Mean Absolute Error ($\text{MAE} = \frac{1}{N} \sum |\hat{y}_i - y_i|$), penalizing predictions proportional to their distance on the ordinal scale.

---

## 3. Strict Input Construction & Leakage Safeguards

Unlike self-supervised pretraining (which used raw state text only), downstream decision prediction legitimately uses the question instructions and option descriptions. To prevent data leakage:
1. **Zero Ground-Truth Contamination:** The input context never includes the target label, correct answer key, or any derived target indicator.
2. **Standardized Context Prompt:**
   ```
   State: <environment_state_text>
   Question / Proposition: <instructions>
   Candidate Options / Scale: <formatted_candidate_descriptions>
   ```
3. **Segregated Targets:** Ground-truth labels are stored strictly in separate integer tensors (`target_idx`).

---

## 4. Multi-Question Handling & Grouped Subset Policy

Many rows contain multiple decision questions about the same state (average 1.91 questions per state). If questions were sampled independently, questions from the same state could leak between training subsets or data splits.

### Subset Sampling Guarantee:
- Fractions ($10\%, 25\%, 50\%, 100\%$) are defined over **unique state groups**, not independent questions.
- All decision questions associated with a selected state group are included.
- Subsets are strictly nested: $S_{10\%} \subset S_{25\%} \subset S_{50\%} \subset S_{100\%}$.
- The identical state subsets are evaluated across all three candidate models (Supervised, JEPA, and MLM).
