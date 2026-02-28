# R-RECP Framework — Implementation-Ready Technical Whitepaper (Corrected)

Document ID: R-RECP-STACK-SPEC-v1.0
Status: Ready for implementation
Scope: Identity stability metrics + semantic field evolution, aligned to CAIROS Visualizer stack
Doctrine: REPA fail-closed, deterministic logging

## 1) Scope & Stack Alignment

This specification maps to existing components at module level (proprietary implementation details omitted):

Simulation engine:
- Simulation core (proprietary)
- Field construction module
- Evolution engine (CPU)
- GPU compute module
- Metrics/centroid module
- Run logging module

Metrics runner (Track A):
- Python reference (Appendix A); CAIROS Daemon open source
- Implements: ICS, API, ECR, LDI, SRV, TCDM
- Logs: SHA-256 of inputs/outputs, SecureExperimentLogger events

Security:
- SecureExperimentLogger v2 (C/C++; in-house)
- SecureLogger core (in-house; path omitted)

## 2) Mathematical Model (Corrected)

### 2.1 Complex Semantic Wavefunction (Formal)

Psi(x,t) = A(x,t) * exp(i * phi(x,t))

Operational definitions:
- x = coordinate in embedding-space grid
- t = discrete turn index (or simulation step)
- A(x,t) = semantic intensity (identity coherence proxy)
- phi(x,t) = phase (semantic drift direction proxy)

Stack mapping (current):
- A(x,0) is the Gaussian field built from coordinates (amplitude field builder)
- phi(x,0) is computed from centroid-relative angles (phase field builder)

### 2.2 Schrodinger-like Evolution (Corrected)

Reference form (phenomenological):

i*hbar * dPsi/dt = [ -(hbar^2 / (2m)) * Laplacian + V_anchor(x,t) + V_entropy(x,t) ] * Psi

Stack mapping (current):

Drift/legacy (Fokker-Planck proxy, REPA baseline):
psi_new = psi + dt * ( D * Laplacian(psi) + Drift )

- D = entropy
- Drift = legacy or full-divergence (see 2.4)

Complex mode (real‑time Schrödinger, implemented):
 i*hbar * dPsi/dt = [ -(hbar^2 / (2m)) * Laplacian + V_anchor + V_entropy ] * Psi

Schemes: Euler, Crank–Nicolson (2D/3D), Split‑step FFT (2D only).

### 2.3 Anchor Potential (Operational)

V_anchor(x) = - alpha * sum_i exp(-||x - a_i||^2 / (2*sigma^2))

- alpha = anchor_strength (manual or legacy=1.0)
- sigma = potential_sigma

Stack mapping:
- Anchor potential builder (proprietary)

### 2.4 Drift Term Definitions (Locked)

Two modes exist in CPU & GPU code:

Legacy (default, REPA baseline):
- Drift = - psi * Laplacian(V)

Full divergence (optional):
- Drift = - psi * Laplacian(V) - (grad(psi) dot grad(V))

Manual scale (optional):
- Drift_Manual = Drift_Legacy * drift_manual_scale

Stack mapping:
- Drift term implementation (legacy / full / manual mode)
- GPU kernel (drift modes 0/1/2; proprietary)

## 3) Thermodynamic Analogy (Operationalized)

Free energy functional (analogy):
- F = U - T * S

Operational definitions (stack-implementable):
- U: internal coherence = mean pairwise cosine similarity (ICS)
- T: semantic temperature = rolling mean emotional intensity score (external)
- S: entropy proxy = Shannon entropy over token/word distribution or embedding variance

Note: T and semantic intensity are not yet computed in engine; they are defined for downstream integration.

## 4) Density Matrix (Future Work)

rho = sum_i p_i |Psi_i><Psi_i|
S_n = -Tr(rho log rho)

Status: Not implemented in current stack. Requires multi-trajectory or ensemble states.

## 5) Axiomatic Confidence Decay (Future Work)

mu(t + dt) = mu(t) * (1 - alpha * dS/dt), alpha > 0

Status: Not implemented. Can be driven by entropy_collapse_rate if desired.

## 6) Metric System (Operational, Implementable)

### 6.1 Identity Consistency Score (ICS)

Implemented in Track A metrics runner:

ICS = (2 / (n(n-1))) * sum_{i<j} cos(e_i, e_j)

Stack mapping: identity_consistency_pairwise()

Interpretation:
- ICS >= 0.85 stable
- ICS <= 0.50 drift

### 6.2 Anchor Persistence Index (API)

API = (1 / (kN)) * sum_i sum_j delta( f(a_i, m_j) > 0 )

Stack mapping: anchor_persistence_index()

### 6.3 Entropy Collapse Rate (ECR)

ECR = 1 - (H_n / H_1)

Stack mapping: entropy_collapse_rate()

### 6.4 Loop Divergence Index (LDI)

LDI = (2 / (n(n-1))) * sum_{i<j} (1 - cos(v_i, v_j))

Stack mapping: loop_divergence_index()

### 6.5 Signal Recursion Variance (SRV)

SRV = Var( 1 - cos(e_t, e_{t-k}) )

Stack mapping: signal_recursion_variance()

### 6.6 Token/Character Drift Metric (TCDM)

- Variance: token_to_word_variance()
- Welch t-test (optional): t_test_ratio_variance()

### 6.7 SDA / HMV / ASV (Unavailable without logits)

- SDA/HMV/ASV require logits and token probabilities.
- Current runner returns null + reason (REPA-compliant).

## 7) Data Flow (Stack-Linked)

1. Conversation file -> turns (parse_conversation)
2. Embeddings via Granite (run_granite)
3. Metrics computed (compute_metrics)
4. SecureExperimentLogger (track_a_start, track_a_metrics_written)
5. Output JSON (RECP-EXP-<run>_metrics.json)

## 8) Security & REPA (Operational)

- Fail-closed if SecureExperimentLogger key is missing.
- All input/output files hashed (SHA-256).
- metrics.json includes security block with:
  - input hashes
  - output hashes
  - logger state
  - explicit notes on unavailable metrics
- Run manifest includes spec/whitepaper path + SHA‑256; missing spec file fails closed.

## 9) Implementation Checklist (Direct Mapping)

Already implemented:
- Fokker-Planck drift + diffusion
- Anchor potential
- Deterministic CUDA kernel
- Metrics: ICS/API/ECR/LDI/SRV/TCDM (Track A runner)
- SecureExperimentLogger integration in metrics runner

Not implemented (requires new work):
- Axiomatic confidence decay
- SDA/HMV/ASV (requires logits/probs)

Implemented (operational, reduced form):
- Density matrix (reduced 2‑D slice) + von Neumann entropy (`density_matrix.csv` + summary)
- Complex mode (real‑time Schrödinger) with Euler/CN (2D/3D) and split‑step (2D).
- Complex diagnostics: fidelity, phase variance, singularity count, coherence length (per‑step logs).

## 10) Corrected Equations (Copy-Ready Block)

Psi(x,t) = A(x,t) * exp(i * phi(x,t))

i*hbar * dPsi/dt = [ -(hbar^2/(2m)) * Laplacian + V_anchor(x,t) + V_entropy(x,t) ] * Psi

F = U - T * S

rho = sum_i p_i |Psi_i><Psi_i| ; S_n = -Tr(rho log rho)

ICS = (2 / (n(n-1))) * sum_{i<j} cos(e_i, e_j)

API = (1 / (kN)) * sum_i sum_j delta( f(a_i, m_j) > 0 )

ECR = 1 - (H_n / H_1)

LDI = (2 / (n(n-1))) * sum_{i<j} (1 - cos(v_i, v_j))

SRV = Var( 1 - cos(e_t, e_{t-k}) )

## Appendix A — Metric Runner Reference

Use the provided Track A Python script as the reference implementation. All function names correspond to the operational definitions above.

---
Signed: Codex (Lead Engineer)
Date: 2026-02-18
