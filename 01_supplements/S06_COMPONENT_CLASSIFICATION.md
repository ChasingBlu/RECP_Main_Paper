# Component classification — ChasingBlu CAIROS ecosystem

**Date:** 2026-02-28  
**Purpose:** Authoritative classification of CAIROS-related software for publication, audit, and distribution.

---

## Proprietary

**CAIROS Visualizer** and **all its modules** are **proprietary**. No part of the Visualizer (Cor, Polaris, Australis, SESHAT, ThermoMap, tooling, or any module) is publicly distributed or open source. Documentation may be shared under confidentiality for review or audit only, as authorized by ChasingBlu R&D Laboratories.

| Component | Classification | Notes |
|-----------|----------------|--------|
| CAIROS Visualizer (Cor, Polaris, Australis, SESHAT, ThermoMap, all modules) | **Proprietary** | Closed source; not for public use or distribution. |

---

## In-house verification layers (not for public use)

The following are **in-house** tools used for **official audits** under the **REDWAX protocol**. They are **not for public use**, **not distributed**, and **not open source**. They are used by ChasingBlu R&D for evidence integrity, model attestation, and document authentication in the context of REPA-aligned audits and publication packages.

| Component | Role | Notes |
|-----------|------|--------|
| **SecureExperimentLogger** (SecureLogger v2.0) | Secure run logging; hash chain; REPA audit trail | Private protocol; not included in any public repo. |
| **Anubis** | Forensic and ascension scanning of neural model weights | Model attestation; report ties evidence to artifact (e.g. RECP publication). |
| **CBDA** (ChasingBlu Document Authenticator; **stealth_c** production implementation) | Document serialization, tamper-evident watermarking | Used to authenticate publication documents and evidence; keys and binaries not for public use. |

**REDWAX protocol:** Internal ChasingBlu process for official audits, evidence lock, and verification layers. Use of SecureLogger, Anubis, and CBDA in that context does not constitute public release of those tools.

---

## Open source

| Component | Location | Notes |
|-----------|----------|--------|
| **CAIROS Daemon** | [https://github.com/ChasingBlu/CAIROS_Daemon](https://github.com/ChasingBlu/CAIROS_Daemon.git) | Python/C++ embedding pipeline; Track A/B; 2D–3D vector-coordinates converter; IEEE SDD in repo. Model weights not included. |

---

## Summary

- **Proprietary:** CAIROS Visualizer and all its modules.  
- **In-house (audit/verification, not for public use):** SecureExperimentLogger, Anubis, CBDA. Used in official audits (REDWAX).  
- **Open source:** CAIROS Daemon (GitHub link above).
