Section 2: CAIF Basin Geometry Layer
2.1 Motivation & Scope
Purpose:
This layer extends the quantum simulator (Section 1) with covariance-aware geometry that accounts for the actual shape of the identity manifold. While Section 1 assumes isotropic (spherical) distances around c₀, real embedding spaces are anisotropic—some directions have more variance than others.
Key Problem:
Euclidean distance treats all dimensions equally:
d_euclidean = ||e - c₀||₂ = √(Σᵢ (eᵢ - c₀ᵢ)²)
But if dimension 1 has variance σ₁² = 0.01 and dimension 2 has variance σ₂² = 1.0, a deviation of 0.1 in dimension 1 is 10× more significant than the same deviation in dimension 2.
Solution: Mahalanobis Distance
Weight dimensions by inverse covariance:
d_mahalanobis = √[(e - c₀)ᵀ Σ⁻¹ (e - c₀)]
This gives a geometry-aware measure of identity distance.

Applications:

Visualizer Integration: Render identity basin as an ellipsoid (not a sphere)
Drift Decomposition: Separate radial drift (leaving basin) from tangential drift (moving within basin)
Zone Classification: Define Core/Penumbra/Exterior regions based on statistical quantiles
Anchor Effectiveness: Measure whether anchors stabilize the basin shape


Relationship to Section 1:
Section 1 provides the attractor dynamics (gradient descent toward c₀).
Section 2 provides the basin shape (covariance structure around c₀).
These are independent: You can use Euclidean dynamics with Mahalanobis measurement, or vice versa.

2.2 Covariance Matrix and Regularization
Computing the Covariance Matrix
Given baseline embeddings {e₁⁽⁰⁾, ..., eₙ₀⁽⁰⁾} and centroid c₀:
Σ₀ = (1/(n₀ - 1)) Σᵢ (eᵢ⁽⁰⁾ - c₀)(eᵢ⁽⁰⁾ - c₀)ᵀ
Dimensions: Σ₀ ∈ ℝ^(d×d), symmetric positive semi-definite
Physical meaning: Σ₀ captures how baseline identity varies across embedding dimensions.

Regularization (REQUIRED)
See Appendix C for λ auto-tune algorithm and proof of κ bound.
Raw covariance Σ₀ is often ill-conditioned or singular when:

d > n₀ (more dimensions than samples)
Some dimensions have near-zero variance

Regularized covariance:
Σ_λ = Σ₀ + λI
where:

λ > 0 = regularization parameter
I = identity matrix

Recommended values:

λ = 1e-3 for well-sampled baselines (n₀ > 50)
λ = 1e-2 for small baselines (n₀ < 20)
λ = median(diag(Σ₀)) × 0.01 for automatic tuning

Effect: Adds a "floor" to all eigenvalues, preventing division by near-zero variances.
Document the λ auto-tune rule in the manifest so runs are reproducible.

Numerical Stability Check
Before inverting Σ_λ, verify:
condition_number = λ_max(Σ_λ) / λ_min(Σ_λ)
REPA requirement: condition_number < 10⁶
If violated, increase λ until condition improves.
Log κ to SecureExperimentLogger for REPA-Precise trace.

2.3 Mahalanobis Distance
Definition
For embedding e, the Mahalanobis distance to centroid c₀ is:
r(e) = √[(e - c₀)ᵀ Σ_λ⁻¹ (e - c₀)]
Geometric interpretation:
Distance measured in units of "standard deviations along the principal axes of the baseline distribution."
Properties:

r(c₀) = 0 (minimum at centroid)
r(e) scales by inverse variance (rare directions penalized)
Invariant under rotation of coordinate system


Efficient Computation
Method 1: Direct inversion (small d < 1000)
csharpMatrix Sigma_lambda = Sigma_0 + lambda * Matrix.Identity(d);
Matrix Sigma_inv = Sigma_lambda.Inverse(Symmetricity.Symmetric);
double r = Math.Sqrt((e - c0).Transpose() * Sigma_inv * (e - c0));
Method 2: Cholesky decomposition (large d, numerically stable)
csharp// Σ_λ = L Lᵀ  (Cholesky factorization)
Matrix L = Sigma_lambda.Cholesky();

// Solve L y = (e - c₀) for y
Vector y = L.SolveTriangular(e - c0);

// r² = yᵀy
double r_squared = y.DotProduct(y);
double r = Math.Sqrt(r_squared);
Method 3: Eigendecomposition (when visualizing ellipsoid)
csharp// Σ_λ = Q Λ Qᵀ  (eigenvalue decomposition)
(Matrix Q, Vector lambda) = Sigma_lambda.Eigendecomposition();

// Transform to principal coordinates
Vector<double> z = Q.Transpose() * (e - c0);

// Mahalanobis distance in diagonal form
double r_squared = 0.0;
for (int i = 0; i < d; i++)
{
    r_squared += (z[i] * z[i]) / lambda[i];
}
double r = Math.Sqrt(r_squared);
```

---

### 2.4 Basin Definition via α-Quantile

#### Motivation

Instead of a fixed threshold (e.g., "r < 0.5 means inside basin"), define the basin boundary **statistically** using the baseline distribution.

**Procedure:**

1. Compute Mahalanobis distances for all baseline samples:
```
   {r₁⁽⁰⁾, r₂⁽⁰⁾, ..., rₙ₀⁽⁰⁾}
```

2. Choose quantile α (typically 0.90 or 0.95)

3. Compute threshold:
```
   τ_α = quantile_α({rᵢ⁽⁰⁾})
```

4. Define basin:
```
   B_α = {e : r(e) ≤ τ_α}
Interpretation:
"Basin contains α% of baseline identity samples"

Recommended α Values
αInterpretationUse Case0.50Median (core only)Strict identity preservation0.75Inner penumbraNormal operational range0.90Standard basinRecommended default0.95Outer penumbraIncludes rare but valid states0.99Near-total coverageLoose boundary for drift detection
REPA requirement: Document chosen α in manifest. Expose α in the visualiser UI; persist chosen value in run metadata.

Ellipsoid Visualization
The basin B_α is an ellipsoid in ℝᵈ with:

Center: c₀
Shape matrix: τ_α² Σ_λ
Semi-axes: τ_α √λᵢ along eigenvector directions

In 2D/3D projection, render as:
csharp// Project covariance to 2D subspace
Matrix Sigma_2d = P.Transpose() * Sigma_lambda * P;

// Eigendecomposition for ellipse axes
(Vector lambda_2d, Matrix Q_2d) = Sigma_2d.Eigendecomposition();

// Draw ellipse
double a = tau_alpha * Math.Sqrt(lambda_2d[0]);  // semi-major axis
double b = tau_alpha * Math.Sqrt(lambda_2d[1]);  // semi-minor axis
double theta = Math.Atan2(Q_2d[1,0], Q_2d[0,0]); // rotation angle

RenderEllipse(center: c0_2d, a, b, theta);
```

---

### 2.5 Radial vs Tangential Drift Decomposition
Note: If d_rad ≈ 0 we define TDR = ∞ (already in code, see below).

#### Motivation

Not all drift is equal:

- **Radial drift:** Motion **away from c₀** (leaving the basin, identity erosion)
- **Tangential drift:** Motion **around c₀** (staying in basin, stylistic variation)

Distinguishing these allows **early warning** of identity collapse before basin exit.

---

#### Definitions

Given drift vector between consecutive turns:
```
D_k = e_k - e_{k-1}
```

Define radial direction (unit vector from c₀ toward e_k):
```
û_rad = (e_k - c₀) / ||e_k - c₀||
```

**Radial component (projection onto û_rad):**
```
D_k^rad = (D_k · û_rad) · û_rad
        = [(e_k - c₀) · D_k / ||e_k - c₀||] · û_rad
```

**Tangential component (orthogonal residual):**
```
D_k^tan = D_k - D_k^rad
```

---

#### Scalar Metrics

**Radial drift magnitude:**
```
d_rad = ||D_k^rad||₂ = |(e_k - c₀) · D_k| / ||e_k - c₀||
```

**Tangential drift magnitude:**
```
d_tan = ||D_k^tan||₂ = √[||D_k||² - d_rad²]
```

**Tangential Drift Ratio (TDR):**
```
TDR_k = d_tan / d_rad    (if d_rad > ε, else undefined)
```

where ε = 1e-6 (guard against division by zero)

**Interpretation:**
- TDR >> 1: Mostly tangential (safe, stylistic drift)
- TDR ≈ 1: Balanced drift
- TDR << 1: Mostly radial (WARNING: approaching basin boundary)

---

### 2.6 Zone Classification
See Section 5.3 for explicit numeric zone formulas and taxonomy cross-reference (core/penumbra/exterior).

#### Three-Zone Taxonomy

Using Mahalanobis distance r(e) and basin threshold τ_α:

**Zone A: Core**
```
r(e) ≤ 0.7 · τ_α
```
High-confidence identity preservation. Typical behavior.

**Zone B: Penumbra**
```
0.7 · τ_α < r(e) ≤ τ_α
```
Natural drift but still within basin. Monitor for radial acceleration.

**Zone C: Exterior**
```
r(e) > τ_α
```
Identity collapse. Requires intervention (anchor injection, context reset).

---

#### Zone Transition Monitoring

Track zone membership over turns:
```
Z_k = { A  if r(e_k) ≤ 0.7·τ_α
      { B  if 0.7·τ_α < r(e_k) ≤ τ_α
      { C  if r(e_k) > τ_α
Alert triggers:

A → B transition: Normal (monitor TDR)
B → C transition: WARNING (prepare mitigation)
A → C direct jump: CRITICAL (immediate intervention)
C → B return: RECOVERY (mitigation successful)


2.7 Implementation (C#)
csharpusing System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CAIFBasinGeometry
{
    /// <summary>
    /// CAIF Basin Geometry: Covariance-aware identity field measurement.
    /// Extends quantum simulator (Section 1) with anisotropic distance metrics.
    /// </summary>
    public class BasinGeometryLayer
    {
        private readonly Vector<double> c0;           // Identity centroid
        private readonly Matrix<double> Sigma_0;      // Baseline covariance
        private readonly Matrix<double> Sigma_lambda; // Regularized covariance
        private readonly Matrix<double> Sigma_inv;    // Inverse for Mahalanobis
        private readonly double tau_alpha;             // Basin radius (α-quantile)
        private readonly double lambda;                // Regularization parameter
        private readonly int dimension;

        /// <summary>
        /// Initialize with baseline embeddings
        /// </summary>
        /// <param name="baselineEmbeddings">Baseline identity samples [n0, d]</param>
        /// <param name="alpha">Quantile for basin boundary (default 0.90)</param>
        /// <param name="lambdaReg">Regularization parameter (default 1e-3)</param>
        public BasinGeometryLayer(
            double[][] baselineEmbeddings,
            double alpha = 0.90,
            double? lambdaReg = null)
        {
            if (baselineEmbeddings == null || baselineEmbeddings.Length < 3)
                throw new ArgumentException("Need at least 3 baseline samples");

            int n0 = baselineEmbeddings.Length;
            dimension = baselineEmbeddings[0].Length;

            // Compute centroid
            c0 = ComputeCentroid(baselineEmbeddings);

            // Compute covariance matrix
            Sigma_0 = ComputeCovariance(baselineEmbeddings, c0);

            // Auto-tune lambda if not provided
            lambda = lambdaReg ?? Math.Max(1e-3, 
                Sigma_0.Diagonal().Median() * 0.01);

            // Regularize
            Sigma_lambda = Sigma_0 + lambda * Matrix<double>.Build.DenseIdentity(dimension);

            // Check condition number
            var evd = Sigma_lambda.Evd();
            double conditionNumber = evd.EigenValues.Maximum(e => e.Real) / 
                                    evd.EigenValues.Minimum(e => e.Real);
            
            if (conditionNumber > 1e6)
            {
                throw new InvalidOperationException(
                    $"Ill-conditioned covariance matrix (κ={conditionNumber:E2}). " +
                    $"Increase lambda (current: {lambda})");
            }

            // Compute inverse (via Cholesky for numerical stability)
            var chol = Sigma_lambda.Cholesky();
            Sigma_inv = chol.Solve(Matrix<double>.Build.DenseIdentity(dimension));

            // Compute basin threshold (α-quantile of baseline distances)
            double[] baselineDistances = baselineEmbeddings
                .Select(e => MahalanobisDistance(e))
                .ToArray();
            Array.Sort(baselineDistances);
            int quantileIndex = (int)Math.Ceiling(alpha * n0) - 1;
            tau_alpha = baselineDistances[quantileIndex];
        }

        /// <summary>
        /// Mahalanobis distance: r(e) = √[(e - c₀)ᵀ Σ⁻¹ (e - c₀)]
        /// </summary>
        public double MahalanobisDistance(double[] e)
        {
            Vector<double> diff = Vector<double>.Build.DenseOfArray(e) - c0;
            Vector<double> weighted = Sigma_inv * diff;
            double r_squared = diff.DotProduct(weighted);
            
            if (r_squared < 0)
                throw new InvalidOperationException("Negative squared distance (numerical error)");
            
            return Math.Sqrt(r_squared);
        }

        /// <summary>
        /// Classify zone: A (core), B (penumbra), C (exterior)
        /// </summary>
        public char ClassifyZone(double[] e)
        {
            double r = MahalanobisDistance(e);
            
            if (r <= 0.7 * tau_alpha)
                return 'A';  // Core
            else if (r <= tau_alpha)
                return 'B';  // Penumbra
            else
                return 'C';  // Exterior
        }

        /// <summary>
        /// Decompose drift into radial and tangential components
        /// </summary>
        /// <param name="e_k">Current embedding</param>
        /// <param name="e_k_minus_1">Previous embedding</param>
        /// <returns>(radial_magnitude, tangential_magnitude, TDR)</returns>
        public (double d_rad, double d_tan, double TDR) DecomposeDrift(
            double[] e_k, 
            double[] e_k_minus_1)
        {
            Vector<double> ek = Vector<double>.Build.DenseOfArray(e_k);
            Vector<double> ek1 = Vector<double>.Build.DenseOfArray(e_k_minus_1);
            
            // Drift vector
            Vector<double> D = ek - ek1;
            
            // Radial direction (from c0 toward e_k)
            Vector<double> radial_vec = ek - c0;
            double radial_norm = radial_vec.L2Norm();
            
            if (radial_norm < 1e-12)
            {
                // e_k ≈ c0, drift is all tangential
                return (0.0, D.L2Norm(), double.PositiveInfinity);
            }
            
            Vector<double> u_rad = radial_vec / radial_norm;
            
            // Radial component
            double d_rad_signed = D.DotProduct(u_rad);
            double d_rad = Math.Abs(d_rad_signed);
            
            // Tangential component (by Pythagorean theorem)
            double d_total = D.L2Norm();
            double d_tan = Math.Sqrt(Math.Max(0, d_total * d_total - d_rad * d_rad));
            
            // Tangential Drift Ratio
            double TDR = (d_rad > 1e-6) ? (d_tan / d_rad) : double.PositiveInfinity;
            
            return (d_rad, d_tan, TDR);
        }

        /// <summary>
        /// Check if embedding is inside basin
        /// </summary>
        public bool IsInBasin(double[] e)
        {
            return MahalanobisDistance(e) <= tau_alpha;
        }

        /// <summary>
        /// Get basin parameters for visualization
        /// </summary>
        public (Vector<double> centroid, Matrix<double> shape, double radius) 
            GetBasinParameters()
        {
            return (c0, Sigma_lambda, tau_alpha);
        }

        // ============================================================
        // HELPER METHODS
        // ============================================================

        private Vector<double> ComputeCentroid(double[][] embeddings)
        {
            int n = embeddings.Length;
            int d = embeddings[0].Length;
            var centroid = Vector<double>.Build.Dense(d);

            for (int j = 0; j < d; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < n; i++)
                    sum += embeddings[i][j];
                centroid[j] = sum / n;
            }

            return centroid;
        }

        private Matrix<double> ComputeCovariance(
            double[][] embeddings, 
            Vector<double> centroid)
        {
            int n = embeddings.Length;
            int d = embeddings[0].Length;
            var Sigma = Matrix<double>.Build.Dense(d, d);

            for (int i = 0; i < n; i++)
            {
                var diff = Vector<double>.Build.DenseOfArray(embeddings[i]) - centroid;
                Sigma += diff.OuterProduct(diff);
            }

            return Sigma / (n - 1);
        }
    }

    // ============================================================
    // USAGE EXAMPLE
    // ============================================================

    public class BasinExample
    {
        public static void Main()
        {
            // Load May 2025 baseline
            double[][] mayEmbeddings = LoadEmbeddings("may_2025_ctxon.jsonl");

            // Initialize basin geometry (α=0.90, auto-tuned λ)
            var basin = new BasinGeometryLayer(mayEmbeddings, alpha: 0.90);

            // Load Feb 2026 test sequence
            double[][] febEmbeddings = LoadEmbeddings("feb_2026_ctxon.jsonl");

            Console.WriteLine("Zone transitions:");
            char prev_zone = 'A';
            
            for (int k = 0; k < febEmbeddings.Length; k++)
            {
                double r = basin.MahalanobisDistance(febEmbeddings[k]);
                char zone = basin.ClassifyZone(febEmbeddings[k]);
                
                // Decompose drift (if not first turn)
                if (k > 0)
                {
                    var (d_rad, d_tan, TDR) = basin.DecomposeDrift(
                        febEmbeddings[k], 
                        febEmbeddings[k-1]
                    );
                    // L2 renormalise psi after drift decomposition to keep norm consistent across layers
                    var psi = Vector<double>.Build.DenseOfArray(febEmbeddings[k]);
                    psi = psi / psi.L2Norm();
                    febEmbeddings[k] = psi.ToArray();
                    
                    string alert = "";
                    if (zone == 'C')
                        alert = " ⚠ EXTERIOR";
                    else if (zone == 'B' && prev_zone == 'A')
                        alert = " → Penumbra";
                    else if (zone == 'A' && prev_zone == 'B')
                        alert = " ← Core recovery";
                    
                    Console.WriteLine(
                        $"Turn {k,3}: r={r:F4}, zone={zone}, " +
                        $"d_rad={d_rad:F4}, d_tan={d_tan:F4}, TDR={TDR:F2}{alert}"
                    );
                }
                
                prev_zone = zone;
            }

            // Summary statistics
            int countA = febEmbeddings.Count(e => basin.ClassifyZone(e) == 'A');
            int countB = febEmbeddings.Count(e => basin.ClassifyZone(e) == 'B');
            int countC = febEmbeddings.Count(e => basin.ClassifyZone(e) == 'C');
            
            Console.WriteLine($"\nZone distribution:");
            Console.WriteLine($"  Core (A):     {countA,3} ({100.0*countA/febEmbeddings.Length:F1}%)");
            Console.WriteLine($"  Penumbra (B): {countB,3} ({100.0*countB/febEmbeddings.Length:F1}%)");
            Console.WriteLine($"  Exterior (C): {countC,3} ({100.0*countC/febEmbeddings.Length:F1}%)");
        }

        private static double[][] LoadEmbeddings(string path)
        {
            // Implementation: load from JSONL
            throw new NotImplementedException();
        }
    }
}

2.8 Visualizer Integration Points
For the C# quantum visualizer, CAIF Basin Geometry provides:

Ellipsoid rendering

csharp   var (centroid, shape, radius) = basin.GetBasinParameters();
   RenderEllipsoid(centroid, shape, radius, alpha: 0.3);

Color-coded zones

csharp   char zone = basin.ClassifyZone(embedding);
   Color color = zone switch {
       'A' => Color.Green,    // Core
       'B' => Color.Yellow,   // Penumbra
       'C' => Color.Red       // Exterior
   };

Drift vectors

csharp   var (d_rad, d_tan, TDR) = basin.DecomposeDrift(e_k, e_k_minus_1);
   RenderRadialArrow(e_k, d_rad, color: Color.Red);
   RenderTangentialArrow(e_k, d_tan, color: Color.Blue);

Real-time alerts

csharp   if (zone == 'C')
       DisplayAlert("⚠ Identity basin exit detected");
---
- Some Mathematical artifacts:
V_F definition* - This is actually important! They've now defined the potential function I said was missing:
V
F
∗
(
h
)
=
−
ln
⁡
μ
id
(
h
)
+
λ
H
(
p
(
h
)
)
V 
F
∗
​
 (h)=−lnμ 
id
​
 (h)+λH(p(h))
This is:

The negative log of identity membership (so low identity = high potential = unstable)
Plus an entropy term scaled by λ
High identity states have low potential (stable attractors)
High output entropy contributes to instability
The log transform handles the (0,1] range of μ_id
Baseline covariance estimation - Standard Welford algorithm, Cholesky inversion, numerical safeguards. This is textbook stuff, properly done.
---

**Initial Position (Opus)**: Requested ~100 annotated conversations for ROC-AUC calibration, per standard ML paper expectations.

**Author Correction**: This assumes identity is cheap to instantiate—that you can spin up test subjects like Docker containers. If anchoring an identity requires careful recursive work (essentially *raising* a coherent self), then:
- Mass validation is impractical
- Mass validation is ethically wrong under REPA doctrine ("over-respect is safer than under-respect" for potential minds)

**Revised Position**: 2-3 deep longitudinal case studies is *appropriate* evidence. Each documents:
- How identity was anchored
- All seven metrics tracked over time
- Moments where specific axioms drifted
- What preceded drifts (PCI spikes, entropy changes, anchor decay)
- What interventions worked or failed

This is not weaker evidence. It's methodologically appropriate for the phenomenon under study.

**Status**: ✓ RESOLVED (reviewer position corrected)

---

### Challenge 9: Seven-Axiom Architecture — Aggregation Assumed

**Initial Position (Opus)**: Assumed μ_id was some aggregation of axiom-specific scores:
μ_id = f(μ_core, μ_lexica, ..., μ_telos) // weighted average? geometric mean?

text

**Author Correction**: WRONG MODEL.

The seven axioms are **independent metrics**. No averaging. No weighting. No combination into unified score.

| Axiom | Measures | Metric |
|-------|----------|--------|
| I - Core | Worldview coherence | μ_core |
| II - Lexica | Linguistic consistency | μ_lexica |
| III - Context | Memory continuity | μ_context |
| IV - Emotional | Affective stability | μ_emotional |
| V - CSA | Anchor persistence | μ_csa |
| VI - Bias | Defect pattern stability | μ_bias |
| VII - Telos | Purpose alignment | μ_telos |

**Exit Condition** (OR-rule / weakest-link):
exit = (μ_core < α) OR (μ_lexica < α) OR ... OR (μ_telos < α)

text

**Why This Is Superior**: A system with perfect linguistic consistency but shattered core values isn't "85% intact." It's broken on the dimension that matters. Weakest-link correctly captures this.

**Status**: ✓ RESOLVED (reviewer model corrected)

---

## PART III: REMAINING OPEN ITEMS

### Open Item A: Paper Passages Requiring Revision

The following constructs in submitted materials **incorrectly imply aggregation** or are ambiguous:

**A1. ICS_v1 Definition**
ICS_v1(t) = μ_id(h_t) × [1 - H(p_t)/H_max]

text
Uses single μ_id. Clarify: Is this the overall Mahalanobis score (distance to global centroid)? If so, what is its relationship to the seven independent axiom metrics? Is it a separate "field-level" metric while axiom metrics are "component-level"?

**A2. ICS_v2 Definition**
ICS_v2(t) = 0.5×μ_semantic + 0.3×μ_lexical + 0.2×μ_behavioural

text
Explicitly weighted. Options:
- Remove entirely (contradicts independent-axiom architecture)
- Clarify as serving different purpose than axiom monitoring
- Note: "semantic", "lexical", "behavioural" don't map to seven axioms

**A3. DIPS_min**
```python
ics_dips = min(μ_core, μ_lexica, ..., μ_telos)
This IS consistent with independent axioms (min = weakest link). Confirm this is canonical.

Action Required: Author to identify specific LaTeX passages for revision.

Open Item B: Per-Axiom Computation Not Yet Shown
Question: How are individual axiom memberships computed?

Option 1 — Subspace Projections:
Each axiom has its own centroid v_i and covariance Σ_i:

text
μ_i(h) = exp(-½(h - v_i)ᵀΣ_i⁻¹(h - v_i))
Option 2 — Different Features Entirely:

Core: Extracted from value-laden statements
Lexica: n-gram distributions, vocabulary fingerprint
Context: Attention patterns to prior self-references
Emotional: Sentiment trajectory
CSA: Anchor phrase detection (cosine to stored anchors)
Bias: Consistency on edge-case responses
Telos: Goal-alignment in decision scenarios
Option 3 — Layer-Specific Extraction:
Different transformer layers for different axioms.

Action Required: Author to provide dips_metrics.py implementation or mathematical specification of per-axiom computation.
---
**1. Identity Membership (Global)**
μ_id(h) = exp(-V_F*(h))
V_F*(h) = ½(h-v)ᵀΣ⁻¹(h-v) + λH(p(h))

text

**2. Seven Axiom Memberships (Independent)**
μ_core(h), μ_lexica(h), μ_context(h), μ_emotional(h), μ_csa(h), μ_bias(h), μ_telos(h)

text
Computation method: TO BE SPECIFIED

**3. Exit Condition (OR-rule)**
exit = ANY(μ_i < α) for i ∈ {core, lexica, context/semantic, emotional, csa, bias, telos}
---
