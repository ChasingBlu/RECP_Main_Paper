# Test Plan — RECP Evidence (Node 01, Node 03, 20260227)

**Document ID:** RECP-TP-001  
**Version:** 1.0  
**Date:** 2026-02-27  
**Scope:** Test objectives, test runs, and pass/fail criteria for the RECP publication evidence package. Aligns with IEEE 29119 (test documentation) and COMPLIANCE_MATRIX_ISO_IEEE.md NC-2.

---

## 1. Purpose

This test plan defines how the evidence supporting the RECP paper (RECP_Paper_Complete_Memorial_20260225.md) is treated as **test runs** and how verification is recorded. It does **not** require new test execution; it documents existing evidence (Node 01, Node 03, 20260227_EVIDENCE_FINAL) as the test baseline and ties them to the paper’s falsification criteria.

---

## 2. Test Objectives

| ID | Objective | Hypothesis |
|----|------------|------------|
| TO-1 | Verify identity basin formation (distance contraction, CIs) | H₁ (identity induction) |
| TO-2 | Verify conservation law compliance (unitarity, energy drift, Hermiticity) | H₂ (quantum simulation) |
| TO-3 | Verify RECP metrics (ICS, API, anchors) and per-axiom CAIF (τ_α, zones) | H₃ (CAIF geometry) |
| TO-4 | Ensure no fabricated or unmeasured values in reported results | REPA / evidence lock |

---

## 3. Test Items and Test Runs

### 3.1 Node 01 — Identity Baseline (Closed-Loop)

| Item | Description | Evidence location |
|------|-------------|-------------------|
| **Test run** | Identity baseline: mean distance to centroid, contraction %, bootstrap CIs | RECP_Publication_Package_20260220/evidence/identity_baseline |
| **Artifacts** | identity_centroid.json, distance distributions, random_gaussian_summary.json, CLAIM_EVIDENCE_TABLE (C1–C10) | Same package |
| **Pass criterion** | Paper Table II values (e.g. 68.4% contraction, 0.1188 vs 0.3763) match artifacts; CIs non-overlapping where claimed | Paper §5.2, §5.3 |
| **Falsification** | Energy fails to decrease monotonically; final state distributions differ from quantum simulation predictions | Paper §3, Layer 1 |

### 3.2 Node 03 — Conservation Audit (CN Hermitian)

| Item | Description | Evidence location |
|------|-------------|-------------------|
| **Test run** | Crank-Nicolson evolution; unitarity, norm, energy drift, Hermiticity | RECP_Publication_Package_20260220/evidence/cn_hermitian_runs_20260222 |
| **Artifacts** | cn_audit_report.json, conservation_report.json (where present), phase evolution frames | Same package |
| **Pass criterion** | Paper Table I values (unitarity 5.37e-14, energy drift 2.37e-13, hermiticity 2.68e-13) match cn_audit_report.json | Paper §5.2 |
| **Falsification** | Conservation laws violated beyond numerical precision (e.g. drift >> 1e-10) | Paper §3 |

### 3.3 20260227_EVIDENCE_FINAL — RECP and CAIF (Lexical / Semantic)

| Item | Description | Evidence location |
|------|-------------|-------------------|
| **Test run** | RECP metrics with anchors (ctx-on/ctx-off); CAIF per-axiom (Lexical, Semantic) | 20260227_EVIDENCE_FINAL |
| **Artifacts** | recp_*.json (anchors_count=13, ICS_anchor_centroid, API), caif_lexical/caif_semantic caif_metrics.json, REPORT_2.4_RECP_vs_CAIF.md | Same directory |
| **Pass criterion** | REPORT_2.4 and paper §5.3.3–5.3.4 match artifacts (τ_α, zone counts A/B/C, mean r, TDR); RECP JSON schema matches SDD §5.2 | Paper §5.3, CAIROS_daemon_SDD_IEEE.md §5.2 |
| **Falsification** | Per-axiom CAIF structure inconsistent with Mahalanobis/zone definitions; RECP metrics not reproducible from same inputs | Paper §3, §6 |

---

## 4. Test Procedures (Summary)

1. **Traceability check:** For each claim in Paper §5 (Results), confirm the cited evidence package and artifact exist and that the reported number appears in that artifact.
2. **Schema check:** For RECP JSON outputs, confirm presence and types of fields in §5.2 of CAIROS_daemon_SDD_IEEE.md (turns_count, anchors_count, metrics.ICS, metrics.ICS_anchor_centroid, metrics.API, etc.).
3. **Falsification criteria:** Document that the falsification criteria in Paper §3 are testable (e.g. “conservation violated beyond 1e-10” can be checked against cn_audit_report.json).

No separate automated test script is required; the test “execution” is the verification that the locked evidence matches the paper.

---

## 5. Test Reporting

| Report | Content | Location |
|--------|---------|----------|
| **Evidence lock** | Claim → artifact mapping | Paper §5 (Evidence lock), COMPLIANCE_MATRIX_ISO_IEEE.md §9 |
| **Run manifests** | Timestamps, hashes, run IDs | run_manifest.json (where present), Appendix B (SHA256) |
| **Test plan** | This document | TEST_PLAN_RECP_Evidence.md |

---

## 6. Compliance Note

This test plan satisfies IEEE 29119 (test documentation) in that:

- Test objectives (TO-1–TO-4) are stated.
- Test items and test runs are identified (Node 01, Node 03, 20260227).
- Pass criteria and falsification criteria are documented.
- Test reporting is via the paper’s Evidence lock and the compliance matrix.

Formal test procedures (step-by-step execution scripts) are not authored; the evidence packages are the test baseline, and verification is by traceability review.

---

## 7. Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-27 | Initial test plan; Node 01, Node 03, 20260227 as test runs. |

---

**End of Test Plan.**
