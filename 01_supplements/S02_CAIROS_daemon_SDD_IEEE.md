## Software Design Description (IEEE 1016 Style)

**System:** CAIROS Daemon (Track B — Primary; Track A — Legacy Reference)

**Document ID:** CAIROS-SDD-TRACK-AB

**Version:** 1.3

**Date:** 2026-02-27

## 1. Introduction

### 1.1 Purpose
This document describes the design of the CAIROS daemon with **Track B (C++/ONNX) as the explicit, primary pipeline**. Track A (Python) is **legacy, reference only**. It covers architecture, components, interfaces, data structures (including the canonical Track B output format), and REPA-aligned operational constraints.

### 1.2 Scope
- **Track B (primary):** C++/ONNX pipeline (embedding + metrics + hidden-state export). Output format is defined explicitly and must match the canonical run outputs (see §5.2).
- **Track A (legacy, reference only):** Python metrics pipeline; not used for production evidence; retained for parity reference.
- SecureLogger v2.0 is referenced but **not included** (private protocol).

### 1.3 Definitions, Acronyms, Abbreviations
- **Track B:** C++/ONNX RECP pipeline; **primary**; output schema defined in §5.2 (canonical reference: `daemon_B/data/LOCKED_TRACK_A_B_Final_20260217/RUN_FEB_TRACK_B/outputs`).
- **Track A:** Python pipeline; **legacy, reference only**; RECP formulas aligned for parity but not the operational output.
- **ICS:** Identity Consistency Score.
- **API:** Anchor Persistence Index.
- **ECR:** Entropy Collapse Rate.
- **LDI:** Loop Divergence Index.
- **SRV:** Signal Recursion Variance.
- **TCDM:** Token-to-word statistical variance (token count vs word count) + Welch t-test.
- **SDA:** Softmax Drift Attribution (logits-dependent).
- **ASV:** Anchor Skew Value (logits-dependent).
- **REPA:** Research Evidence & Provenance Alignment protocol.

## 2. System Overview
The system computes RECP metrics over contextual and non-contextual inputs using IBM Granite embeddings. **Track B (C++/ONNX)** is the operational pipeline: ONNX Runtime embeds turns and anchors; `recp_metrics` computes metrics; output is written in the canonical JSON format (§5.2). Track A (Python) is legacy/reference only.

## 3. Design Considerations
- **Determinism:** No silent fallbacks. Missing inputs or models raise errors.
- **Reproducibility:** When SecureLogger is enabled, inputs and outputs are hashed.
- **Portability:** No local absolute paths or private datasets.
- **Security:** SecureLogger is private; hooks exist but are disabled by default.
- **REPA Alignment:** No fabricated values. Missing dependencies result in `null` outputs and explicit notes.
- **Good Practice (Tooling):** Explicit compiler/toolchain versions are documented; optional components are clearly labeled; CUDA/OpenCV are not required for core metric pipelines.

## 4. Architecture

### 4.1 High-Level Components
- **Track B (primary) — C++** (`native/` in daemon_B or GitHub subset)
  - `granite_embedder` (ONNX + SentencePiece)
  - `recp_metrics` (core metric formulas)
  - CLIs: `recp_metrics_cli`, `granite_embedder_cli`, `granite_hidden_state_cli`, `identity_anchor_cli`, `granite_tokenizer_cli`, `coords_from_embeddings_cli`
- **Track A (legacy, reference only):** Python `track_a_ctxonoff_metrics.py`, Granite loader, `granite_loader.py` / `model_hub.py`

### 4.2 Data Flow (Track B — Primary)
1. Input: turns file (and optionally anchors file); embeddings from `granite_embedder_cli` or precomputed JSONL.
2. `recp_metrics_cli` loads embeddings, computes metrics via `recp_metrics`.
3. Write **exactly one JSON file per run** (ctx-on and ctx-off typically produce two files). Schema: §5.2.

### 4.3 Data Flow (Track A — Legacy Reference)
1. Parse contextual/non-contextual turns; embed via Granite loader.
2. Compute metrics (parity with §7.1); write Track A–specific JSON (`contextual`, `non_contextual`, `security`). Not the canonical output format.

## 5. Data Design

### 5.1 Input Formats
- **Turns:** `turn_id|speaker|text` per line.
- **Anchors:** one anchor per line; Track A also accepts JSON with `anchors` array.

### 5.2 Output Format (Track B — Canonical, Explicit)

**Reference:** `D:\ChasingBlu_RND\Lab\Active\daemon_B\data\LOCKED_TRACK_A_B_Final_20260217\RUN_FEB_TRACK_B\outputs`

Track B `recp_metrics_cli` writes **one JSON object per run** (one file for ctx-on, one for ctx-off). Output MUST match this schema exactly.

**Root object (single line or pretty-printed):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `turns_count` | number | yes | Number of turns. |
| `anchors_count` | number | yes | Number of anchors. |
| `srv_k` | number | yes | Lag used for SRV (e.g. 1). |
| `metrics` | object | yes | All RECP metric values (see below). |
| `availability` | object | yes | Per-metric availability flags for logits-dependent metrics. |
| `embedding_source` | string | yes | Path to the embeddings JSONL used (or empty string if in-memory). |

**`metrics` object — every key present, numeric:**

| Key | Type | Description |
|-----|------|-------------|
| `ICS` | number | Identity Consistency Score (primary: anchor-centroid when anchors exist, else pairwise). |
| `ICS_pairwise` | number | Pairwise ICS (audit). |
| `ICS_anchor_centroid` | number | Anchor-centroid ICS (same as ICS when anchors present). |
| `API` | number | Anchor Persistence Index. |
| `ECR` | number | Entropy Collapse Rate. |
| `LDI` | number | Loop Divergence Index. |
| `TCDM` | number | Token-to-word variance. |
| `SRV` | number | Signal Recursion Variance. |
| `HMV` | number | Hook Mutation Vector (0 when logits unavailable). |
| `RDS` | number | Recursive Drift Score. |
| `MPI` | number | Mutation Pressure Index. |
| `SDA` | number | Softmax Drift Attribution (0 when unavailable). |
| `ASV` | number | Anchor Skew Value (0 when unavailable). |

**`availability` object — booleans for logits-dependent metrics:**

| Key | Type | Description |
|-----|------|-------------|
| `SDA` | boolean | true if SDA was computed from logits. |
| `ASV` | boolean | true if ASV was computed from logits. |

**Example (canonical run):**

```json
{"turns_count":4,"anchors_count":13,"srv_k":1,"metrics":{"ICS":0.75694422,"ICS_pairwise":0.98875115,"ICS_anchor_centroid":0.75694422,"API":0.63461538,"ECR":-0.31765004,"LDI":0.011248853,"TCDM":0.0059264462,"SRV":0.000054396302,"HMV":0,"RDS":0.018824563,"MPI":0.014118422,"SDA":0,"ASV":0},"availability":{"SDA":false,"ASV":false},"embedding_source":"<path_to_embeddings.jsonl>"}
```

**File naming convention (canonical runs):** e.g. `recp_metrics_cli_ctxon_pairwise_pyemb.json`, `recp_metrics_cli_ctxoff_pairwise_pyemb.json` — one file per contextual condition.

### 5.3 Provenance Metadata (Track A — Legacy)
- Track A only: `security.sha256_inputs`, `security.sha256_outputs`, `security.metric_notes`. Not part of Track B output.

## 6. Interface Design

### 6.1 Track B CLIs (Primary)
```
recp_metrics_cli --turns <txt> --output <json> [--anchors <txt>] \
  [--embeddings <jsonl> | --model-dir <dir>] [--secure-log-dir <dir> --secure-key <path>]
```
Output: single JSON file per invocation; schema §5.2. Typical run: two invocations (ctx-on, ctx-off) → two files (e.g. `recp_metrics_cli_ctxon_pairwise_pyemb.json`, `recp_metrics_cli_ctxoff_pairwise_pyemb.json`).

```
granite_embedder_cli --model-dir <dir> --input <txt> --output <jsonl> [--device cuda]
granite_hidden_state_cli --model-dir <dir> --input <txt> --output <jsonl>
identity_anchor_cli ...
coords_from_embeddings_cli --input-root <dir_with_jsonl> --out-dir <out> --dims 3 [--anchors <anchors_jsonl>]
granite_tokenizer_cli ...
```

### 6.2 Track A CLI (Legacy, Reference Only)
```
python python/track_a_ctxonoff_metrics.py \
  --run-dir <dir> --ctx-on <txt> --ctx-off <txt> \
  --anchors <txt_or_json> --model-dir <model_dir>
```
Output format is Track A–specific (`contextual`, `non_contextual`, `security`); not the canonical §5.2 schema.

## 7. Component Design

### 7.1 RECP Metrics — Formal Definitions (Math)

The following formulas match the C++ `recp_metrics` implementation and Track A Python. Cosine uses unnormalized vectors: cos(a,b) = (a·b)/(‖a‖‖b‖).

- **ICS (primary when anchors exist):** Weighted anchor centroid e_id = (Σ w_i e_i)/(Σ w_i) with w_i = 1 − max_{j≠i} cos(e_i, e_j) (anchors); if Σw_i = 0 use uniform weights. Then ICS = (1/T) Σ_t cos(v_t, e_id). When anchors are absent, ICS = ICS_pairwise.
- **ICS_pairwise (audit):** ICS_pairwise = (2/(n(n−1))) Σ_{i<j} cos(v_i, v_j) over turn embeddings.
- **API:** API = (1/(kN)) Σ_i Σ_j δ(turn j contains anchor i). Case-insensitive substring match; k = number of anchors, N = number of turns.
- **ECR:** ECR = 1 − (H_n / H_1). H_t = Shannon entropy (word-level: tokenize on whitespace, empirical distribution, H = −Σ p log₂ p) of turn t. H_1 = first turn, H_n = last turn.
- **LDI:** LDI = (2/(n(n−1))) Σ_{i<j} (1 − cos(v_i, v_j)) over turn embeddings (average pairwise divergence).
- **SRV:** SRV = Var{ 1 − cos(e_t, e_{t−k}) } over t = k+1..N (variance of k-lag drift; k is configurable, e.g. 1).
- **TCDM (twv):** Per-turn ratio r_t = (token_count_t)/(word_count_t). When token counts are provided (e.g. SentencePiece), use model-consistent token counts; else legacy uses character count as proxy. TCDM = (1/|turns|) Σ_t (r_t − r̄)² (variance of ratios across turns). Optional Welch t-test is not implemented in this subset.
- **RDS (recursive drift score):** Mean drift from first embedding: (1/(n−1)) Σ_{i≥1} (1 − cos(v_1, v_i)).
- **MPI (mutation pressure index):** Mean of per-embedding drift from first: same drift values as RDS, MPI = mean(drifts).
- **SDA / ASV:** Require logits or token probabilities; not available in this pipeline; output null with notes.

### 7.2 Track B Metrics (C++ — Primary)
`recp_metrics` implements the formulas in §7.1. Output is written in the canonical schema (§5.2): all fields `turns_count`, `anchors_count`, `srv_k`, `metrics` (ICS, ICS_pairwise, ICS_anchor_centroid, API, ECR, LDI, TCDM, SRV, HMV, RDS, MPI, SDA, ASV), `availability` (SDA, ASV), `embedding_source`. If anchors exist, `ics_anchor` is primary `ics`; `ics_pairwise` is audit.

### 7.3 Track A Metrics (Python — Legacy Reference)
- Same formulas as §7.1; output format is Track A–specific (contextual / non_contextual / security), not §5.2. Retained for parity reference only.

### 7.4 Coords Converter (C++)
- Computes 2D/3D PCA coordinates from RECP JSONL embeddings.
- Uses May embeddings as PCA basis; projects Feb and anchors into the same space.
- Outputs CSV/JSON coords plus PCA metadata (mean/components/minmax).

### 7.5 Tokenizer (C++)
- `granite_tokenizer_cli` produces per-line token counts using SentencePiece from the model directory.
- Output JSONL can be fed into Track A (`--token-counts-ctxon/ctxoff`) or Track B (`--token-counts`).

## 8. Error Handling
- Missing model files or embeddings: hard error.
- Missing anchors: ICS falls back to pairwise; notes indicate anchor absence.
- Missing optional dependencies (e.g., `scipy`): corresponding values are `null` with explicit notes.
- SecureLogger enabled without key: hard error (fail-closed for secure logging).

## 9. Security & REPA Alignment
- SecureLogger v2.0 is **private** and not included in this repository.
- When SecureLogger is not present, the pipeline still computes metrics and marks security fields as unavailable.
- No metric values are fabricated. All omissions are explicit and noted.

## 10. Build & Deployment
See `docs/DEPENDENCIES.md` for versions and `README.md` for build steps.

## 11. Assumptions and Constraints
- IBM Granite-Embedding-278M model is available locally.
- ONNX Runtime and SentencePiece are installed for Track B.
- SecureLogger is optional and external.

## 12. Traceability Matrix (Summary)
- **Canonical output format** → §5.2; reference run: `daemon_B/data/LOCKED_TRACK_A_B_Final_20260217/RUN_FEB_TRACK_B/outputs`.
- **RECP metrics (ICS, API, ECR, LDI, TCDM, SRV, RDS, MPI, etc.)** → Track B `recp_metrics` + §7.1; output keys match `metrics` in §5.2.
- **Provenance (Track A legacy)** → Track A only: `security.sha256_inputs`, `security.sha256_outputs`, `security.metric_notes`.

### 12.1 Requirements → Design → Verification (IEEE 1016)

| Req ID | Requirement (summary) | Design element | Verification / evidence |
|--------|------------------------|----------------|--------------------------|
| REQ-01 | Identity baseline: distance contraction, bootstrap CIs | §4.2 data flow; §7.1 ICS, centroid | Node 01: RECP_Publication_Package_20260220/evidence/identity_baseline; Paper Table II, §5.3 |
| REQ-02 | Conservation: unitarity, energy drift, Hermiticity | Simulation.Core (CN solver); not in this SDD | Node 03: same package, cn_hermitian_runs_20260222/cn_audit_report.json; Paper Table I, §5.2 |
| REQ-03 | RECP metrics + anchors; CAIF per-axiom (Lexical, Semantic) | §4.1–4.2 Track B CLIs; §5.2 output schema; CaifBatch (external) | 20260227_EVIDENCE_FINAL: recp_*.json, caif_*/*.json, REPORT_2.4; Paper §5.3.3–5.3.4 |
| REQ-04 | No fabricated values; missing deps → null + notes | §3 Design considerations; §8 Error handling | Paper §5 Evidence lock; COMPLIANCE_MATRIX_ISO_IEEE.md §9 |
| REQ-05 | Deterministic, reproducible (hash chain when SecureLogger enabled) | §3, §9 Security & REPA | run_manifest.json, Appendix B SHA256; KEY_HANDLING_POLICY.md |

---

## 13. REPA Amendment Log
- 2026-02-17: TCDM discrepancy identified (legacy char/word variance). Amended to model-consistent token counts via granite_tokenizer_cli + token-count overrides in Track A/B.
- 2026-02-27: Track B designated primary; Track A legacy/reference only. Canonical output format (§5.2) aligned to `daemon_B/data/LOCKED_TRACK_A_B_Final_20260217/RUN_FEB_TRACK_B/outputs`.

## 14. SecureLogger (Closed-Source) — Workflow Summary
SecureLogger v2.0 is **closed-source** and **not distributed**. This document provides only a high-level workflow description and **does not disclose implementation details, cryptographic parameters, or code snippets**.

Workflow (high-level):
- **Track B (`recp_metrics_cli`) is fail-closed** when SecureLogger is required; a key + log directory must be provided or the run aborts.
- **Track A (Python)** runs without SecureLogger unless explicitly enabled.
- When SecureLogger is enabled, the pipeline records **tamper-evident hashes** of inputs/outputs and writes an audit trail to a secure log directory.
- Any run performed **without** SecureLogger keys/signatures is considered **external/unverified** and **will not be attributed to ChasingBlu R&D Labs**.

---
**Signed:** Codex — Systems Engineering Assistant