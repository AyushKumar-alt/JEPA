# Dataset Integrity & Provenance Verification Report

## 1. Overview and Investigation Objective
This audit examines the integrity, provenance, schema consistency, and split isolation of the dataset files located in `data/bev-decision/` against the official Hugging Face repository `avbiswas/bev-decision` (default configuration, pinned release corresponding to the 150K mixture).

In previous logs, an OS reading error was encountered on the local `train.parquet` file (`Couldn't deserialize thrift: No more data to read.`), resulting in a backup (`train.parquet.original_backup`) and an Arrow-based reconstruction. This audit systematically investigates:
1. The root cause of the initial deserialization defect.
2. The exact cryptographic hashes and sizes of local files and official remote files.
3. Row-by-row and column-by-column semantic fidelity between reconstructed data and the official repository release.
4. Cross-split isolation between `train.parquet` and `test.parquet`.

---

## 2. File Artifacts, Sizes, and Cryptographic Hashes

### 2.1 File Comparison Matrix

| File Identifier | Location | Size (Bytes) | SHA-256 Checksum | PyArrow Read Status | Row Count |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Official Remote Train** | `avbiswas/bev-decision/data/train.parquet` | 31,742,843 | `1b163182db155ba986c8ab8d3d582ac09ec1b1dcbc13c65d048e61379be9fe5b` | **Valid** | 125,614 |
| **Current Local Train** | `data/bev-decision/train.parquet` | 31,742,843 | `1b163182db155ba986c8ab8d3d582ac09ec1b1dcbc13c65d048e61379be9fe5b` | **Valid** | 125,614 |
| **Original Local Backup** | `data/bev-decision/train.parquet.original_backup` | 57,063,262 | `25ec7e8b8a7622114ba1b5811c25b15f22ae0c34118ab77327f3a9f81ad7deb2` | **Corrupted Thrift** | N/A |
| **Reconstructed Cache** | Local Arrow Export | 50,746,254 | `9b0c4ce50c8c6a7ddde56dbcfce8c14df0e336959b57f2708f2b5b24444d30d9` | **Valid** | 125,614 |
| **Official Remote Test** | `avbiswas/bev-decision/data/test.parquet` | 5,745,752 | `eba9f9df6ad54ccdc3aa6516ab97e4f9dea4030422e0dc90510b85e462206a61` | **Valid** | 24,386 |
| **Current Local Test** | `data/bev-decision/test.parquet` | 5,745,752 | `eba9f9df6ad54ccdc3aa6516ab97e4f9dea4030422e0dc90510b85e462206a61` | **Valid** | 24,386 |

### 2.2 Root-Cause Diagnosis of `train.parquet.original_backup`
Inspection of `train.parquet.original_backup` (size: 57,063,262 bytes) revealed that the file was modified at 12:48:02 on 2026-10-28. Byte inspection showed multiple Thrift file metadata headers concatenated at byte offset 57,062,394, indicating an interrupted file write or concatenated streams during initial workstation staging.

An independent download of `data/train.parquet` directly from `avbiswas/bev-decision` into a segregated verification directory (`outputs/research_audit/verification_data/data/train.parquet`) downloaded cleanly at 31,742,843 bytes.

---

## 3. Row-by-Row Semantic Integrity Verification

To ensure zero divergence between the reconstructed dataset used in prior preliminary runs and the official repository release, an exhaustive row-by-row comparison was conducted across all 125,614 rows:

```python
# Verification script excerpt:
# Schema comparison:
assert official_train.schema.equals(local_train.schema)
# Column equality across all rows:
for col in ['state', 'questions_json', 'domain', 'question_types', 'question_count']:
    assert official_train[col].to_pylist() == local_train[col].to_pylist()
```

### Results:
1. **Schema Match:** True (100% type parity: `state: large_string`, `questions_json: large_string`, `domain: string`, `question_types: list<string>`, `question_count: int16`).
2. **`state` parity:** Exactly 125,614 out of 125,614 strings match bit-for-bit (0 mismatches).
3. **`questions_json` parity:** Exactly 125,614 out of 125,614 strings match bit-for-bit (0 mismatches).
4. **`domain` parity:** Exactly 125,614 out of 125,614 strings match bit-for-bit (0 mismatches).
5. **`question_types` parity:** Exactly 125,614 out of 125,614 lists match bit-for-bit (0 mismatches).
6. **`question_count` parity:** Exactly 125,614 out of 125,614 integers match bit-for-bit (0 mismatches).

The official remote file (SHA-256: `1b163182db155ba986c8ab8d3d582ac09ec1b1dcbc13c65d048e61379be9fe5b`) has now been installed at `data/bev-decision/train.parquet`, while `data/bev-decision/train.parquet.original_backup` is safely retained.

---

## 4. Test Split Invariance and Cross-Split Contamination Audit

### 4.1 Test Split Invariance
`data/bev-decision/test.parquet` was verified against the official remote file. Both files share identical size (5,745,752 bytes) and identical SHA-256 hash (`eba9f9df6ad54ccdc3aa6516ab97e4f9dea4030422e0dc90510b85e462206a61`). The local test file was never altered or overwritten.

### 4.2 Cross-Split Normalized State Leakage Check
We extracted all states from both splits, normalized them by lowercasing and stripping whitespace, and computed set intersections:
- **Unique Normalized Training States:** 116,655
- **Unique Normalized Test States:** 23,352
- **Cross-Split State Overlap Count:** **0 (0.00%)**

### Conclusion:
There is zero cross-split state overlap between the official training and test partitions. The dataset meets the integrity gate criteria.
