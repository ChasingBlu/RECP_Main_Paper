# SecureLogger Notice (In-house verification layer)

**SecureExperimentLogger** (SecureLogger v2.0) is an **in-house verification layer** and **not for public use**. It is used in **official audits** under the **REDWAX protocol**. It is private, non-open-source, and **not distributed**. This repository **does not** include its implementation. The Track A/B pipelines (e.g. CAIROS Daemon) expose hooks and optional flags to integrate SecureLogger, but all secure logging is disabled by default in public builds.

If you have access to SecureLogger v2.0 under Lab authorization, place the module on your build/PYTHONPATH and enable `CAIROS_SECURE_LOGGER` (Python) or `-DCAIROS_WITH_SECURE_LOGGER=ON` (CMake). See **COMPONENT_CLASSIFICATION.md** for full component classification (Proprietary / In-house / Open source).
