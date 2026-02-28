# Key Handling Policy — SecureLogger and Evidence Integrity

**Document ID:** RECP-KHP-001  
**Version:** 1.0  
**Date:** 2026-02-27  
**Scope:** SecureLogger v2.0 key storage, access, and lifecycle for RECP evidence runs. Aligns with ISO/IEC 27001 (applicable clauses) and COMPLIANCE_MATRIX_ISO_IEEE.md NC-4.

---

## 1. Purpose

This policy documents how cryptographic keys used for tamper-evident audit trails (SecureLogger v2.0) are generated, stored, and accessed during RECP evidence generation. It does **not** disclose implementation details of SecureLogger (closed-source).

---

## 2. Scope

- **In scope:** Key material used by `recp_metrics_cli` and related pipelines when SecureLogger is enabled (e.g. `--secure-key`, `--secure-log-dir`). Evidence packages (20260227_EVIDENCE_FINAL, RECP_Publication_Package_20260220) produced under this policy.
- **Out of scope:** Application-level API keys, model weights, or third-party credentials.

---

## 3. Key Storage (Current Practice)

| Aspect | Practice | Notes |
|--------|----------|--------|
| **Location** | Key material supplied via file path (e.g. `secure_key.bin`) or equivalent mechanism. | Not embedded in source code or committed to version control. |
| **Access** | Single-operator research environment. Only the principal investigator (or designated operator) has access to the key file and log directory. | No multi-user or role-based key access in current setup. |
| **Generation** | Key generation is performed by SecureLogger tooling or external process; key is written once to a path outside the repository. | No key generation logic in open-source CAIROS Daemon. |
| **Repository** | Keys and secure log contents are **never** committed to the repository. `.gitignore` (or equivalent) excludes key paths and log directories. | Evidence packages may include SHA256 hashes of inputs/outputs; they do not include the key or raw log secrets. |

---

## 4. Key Lifecycle

- **Creation:** Key created once per evidence campaign or per secure log root; not rotated during a single publication package run.
- **Rotation / Revocation:** No operational rotation or revocation is performed for the 20260227 or RECP_Publication_Package_20260220 evidence. For future campaigns, rotation policy to be documented in an amendment to this policy.
- **Destruction:** Key material may be deleted or overwritten after evidence is locked and hashes are published; retention is at operator discretion. Secure log directory is retained for audit until package is finalized.

---

## 5. Higher-Assurance Options (Optional)

For environments requiring stricter controls, the following are **optional** and not currently executed:

- **HSM or hardware security module:** Key generated and stored in HSM; SecureLogger (or wrapper) uses HSM for signing. Not implemented for current evidence.
- **Azure Key Vault / cloud KMS:** Key stored in managed vault; application retrieves key reference at runtime. Not implemented for current evidence.
- **Key ceremony / dual control:** Key generation and loading require two authorized persons. Not implemented for single-operator research.

Documenting these options does not imply compliance with them; they are listed as remediation paths for NC-4 (COMPLIANCE_MATRIX_ISO_IEEE.md).

---

## 6. Compliance Statement

- Current practice satisfies **documented** key handling for single-operator, research-only use.
- Key material is **not** in the repository or in any publicly released artifact.
- This policy is the **evidence** for ISO/IEC 27001–related key management (COMPLIANCE_MATRIX_ISO_IEEE.md §5, NC-4). HSM/Key Vault integration remains optional for higher assurance.

---

## 7. Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-27 | Initial policy; aligns with RECP evidence lock and NC-4. |

---

**End of Key Handling Policy.**
