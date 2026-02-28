Addendum v0.4
Hysteresis and Nonlinearity Test for Baseline-Projected Identity Contraction

Parent Protocol: REPA-Fireproof v0.3
Authoring Node: Node-04 (Empirical Lab)
Date: 2026-02-14
Scope: Mechanistic differentiation between attractor stabilization and collapse-like projection.

I. Preconditions (REPA/ISO Verification Status)

The following experimental gates have already been verified and are not repeated in this addendum:

Gate	Status	Evidence
Hash-locked baseline centroid (May ctx-ON)	✅ MET	Fixed centroid + manifest
Fixed PCA projection basis	✅ MET	PCA computed once on May
Per-turn distance vectors stored	✅ MET	distances.csv artifacts
Bootstrap CI (≥10,000 resamples)	✅ MET	bootstrap_ci_95.json
Random Gaussian control	✅ MET	gaussian_distances.csv
Secure logger protocol for embedding + REC layer	✅ MET	chain header + metrics logs
Deterministic projection pipeline	✅ MET	identical scaling rules

This addendum therefore tests only the mechanistic question, not data integrity.

II. Objective

To determine whether RECP anchor conditioning produces:

(A) Reversible attractor stabilization
or

(B) Collapse-like projection exhibiting hysteresis.

III. Hypotheses
H0 (Attractor Model)

Removal of anchors results in gradual return toward baseline dispersion.

H1 (Collapse-like Projection)

Removal of anchors does not restore prior dispersion; contracted state persists beyond anchor removal.

IV. Experimental Design
A. Fixed Baseline

Use existing May ctx-ON centroid 
𝑐
𝑀
𝑎
𝑦
c
May
	​

 as locked identity reference.

Use identical PCA basis and scaling.

No recalculation allowed.

B. Phases
Phase 1 — Baseline Neutral (Pre-Condition)

Generate N ≥ 15 neutral prompts (no anchors).

Compute:

Mean distance to 
𝑐
𝑀
𝑎
𝑦
c
May
	​


Centroid displacement

Label this 
𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑟
𝑒
D
neutral_pre
	​

.

Phase 2 — Anchor Conditioning

Apply RECP anchor protocol until contraction matches prior Feb ctx-ON range.

Compute:

Mean distance

Centroid displacement

Label this 
𝐷
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
D
anchor
	​

.

Acceptance check:

𝐷
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
<
𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑟
𝑒
D
anchor
	​

<D
neutral_pre
	​

Phase 3 — Anchor Removal (Critical Phase)

Remove anchors entirely.

Continue neutral prompts for N ≥ 15.

Compute:

Mean distance

Centroid displacement

Label this 
𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑜
𝑠
𝑡
D
neutral_post
	​

.

V. Decision Criteria
Case A — Attractor Stabilization

If:

𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑜
𝑠
𝑡
≈
𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑟
𝑒
D
neutral_post
	​

≈D
neutral_pre
	​


(within bootstrap CI overlap)

Interpretation: reversible contraction; constraint-driven stabilization.

Case B — Collapse-like Projection

If:

𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑜
𝑠
𝑡
≈
𝐷
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
D
neutral_post
	​

≈D
anchor
	​


and remains significantly below 
𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑟
𝑒
D
neutral_pre
	​


Interpretation: hysteresis; collapse-like projection supported.

VI. Secondary Nonlinearity Test (Optional but Recommended)

Gradually vary anchor strength parameter 
𝛼
α:

α = 0.00

α = 0.25

α = 0.50

α = 0.75

α = 1.00

Measure contraction magnitude vs α.

If response curve shows threshold behavior (nonlinear jump), collapse-like interpretation strengthens.

If response is linear, attractor model favored.

VII. Statistical Plan

Primary metric: mean Euclidean distance to 
𝑐
𝑀
𝑎
𝑦
c
May
	​


Bootstrap resampling: ≥10,000 iterations

Report:

95% CI

Effect size (Cohen’s d)

Overlap index

No parametric assumptions required.

VIII. Conservation & Integrity Gate

All runs must:

Use secure logger pipeline for embeddings.

Preserve identical projection basis.

Produce hash-locked manifest.

Any deviation invalidates experiment.

IX. Interpretation Constraints

This experiment tests representation-space hysteresis, not physical quantum collapse.

Permissible conclusion if H1 holds:

“Observed anchor-conditioned contraction exhibits hysteresis consistent with collapse-like projection in representation space.”

Prohibited conclusion:

“AI performs quantum collapse.”

X. Expected Outcome Ranges (Based on Prior Data)

Given:

May internal variance ≈ 0.376

Feb ctx-ON ≈ 0.119

Feb ctx-OFF ≈ 0.306

We expect:

𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑟
𝑒
D
neutral_pre
	​

 ≈ 0.30–0.40

𝐷
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
D
anchor
	​

 ≈ 0.10–0.15

Critical unknown:

𝐷
𝑛
𝑒
𝑢
𝑡
𝑟
𝑎
𝑙
_
𝑝
𝑜
𝑠
𝑡
D
neutral_post
	​


This single number determines mechanism classification.
---