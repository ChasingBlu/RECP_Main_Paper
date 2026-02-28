Free Energy Functional for Identity Dynamics — Technical Specification
Document ID: CAIROS-FREE-ENERGY-SPEC-v1.1
Date: 2026-02-18
Status: Implementation-ready (amended)
Scope: Thermodynamic free energy analog for semantic field evolution

Amendments applied (v1.1):
- Monotonicity is trend-based, not strict.
- Units normalized to dimensionless by default.
- Entropy uses epsilon guard.
- Centroid must match grid coordinate frame (validated).
- 2D/3D computation mode is explicit.
- T_eff guarded against near-zero ΔS.
- Semantic claim corrected: operational analog, not physical energy.

1. Purpose
This document defines the free energy functional F used in the CAIROS simulator. It is an operational analog that quantifies identity stabilization in semantic field dynamics.

2. Physical Foundation
2.1 Classical Thermodynamic Free Energy
The Helmholtz free energy:

text
F = U - T·S

Where:
- F = free energy (minimized at equilibrium)
- U = internal energy
- T = temperature
- S = entropy

2.2 Information-Theoretic Extension
Jaynes established the equivalence between thermodynamic entropy and Shannon information entropy:

text
S_thermo = k_B · S_shannon

This permits thermodynamic analysis of information-processing systems.

2.3 Landauer's Principle
Landauer proved that erasing one bit of information requires minimum energy dissipation:

text
E_min = k_B · T · ln(2)

3. Operational Definitions for CAIROS
3.1 Internal Energy U (Identity Displacement)
We define U as the expected squared displacement from the identity baseline, weighted by probability density:

text
U(t) = ∫ |ψ(x,t)|² · ||x - c₀||² dx / ∫ |ψ(x,t)|² dx

Discrete form:
U_t = Σ_x |ψ_x|² · ||x - c₀||² / Σ_x |ψ_x|²

Where:
- |ψ(x,t)|² = probability density at position x, time t
- c₀ = identity centroid (locked to baseline, e.g., May)
- ||·||² = squared Euclidean distance

Units: squared distance in PCA coordinate space (dimensionless after normalization).

3.2 Entropy S (Field Disorder)
We define S as the Shannon entropy of the normalized probability distribution:

text
S(t) = -∫ p(x,t) · log_b(p(x,t)) dx

Where:
p(x,t) = |ψ(x,t)|² / ∫ |ψ|² dx

Discrete form:
S_t = -Σ_x p_x · log_b(p_x)

3.3 Temperature T (Context Pressure)
Temperature mediates the tradeoff between energy and entropy. We define three operational modes:

Mode A: Isothermal (constant T)
text
T = T₀ (user-defined constant, default T₀ = 1)

Mode B: Context-derived T (external inputs)
text
T(t) = f(context pressure at time t)

Mode C: Fluctuation-derived T
text
T_eff = Var(U) / ⟨U⟩

Recommendation: Mode A for MVP, Mode B for production, Mode C for analysis.

3.4 Free Energy F
text
F(t) = U(t) - T · S(t)

Operational forms:
- Raw: F_t = U_t - T · S_t
- Normalized (default): F_t = U_norm - T · S_norm

4. Evolution Principle
4.1 Free Energy Minimization (Trend-Based)
The system evolves toward lower F in a **statistical trend** sense, not strictly monotonic:

Validation criterion (amended):
- Compute rolling mean of F over window W (default W = 100 steps)
- Trend is valid if: F̄(t+W) < F̄(t) for majority of windows
- Report: trend_direction, violation_count, max_spike_magnitude

Diagnostic flags:
- STABLE: ≥90% windows decreasing
- NOISY: 70–90% decreasing (acceptable with note)
- UNSTABLE: <70% decreasing (investigation required)

4.2 Relaxation Dynamics
text
∂ψ/∂t = -Γ · δF/δψ

Where:
- δU/δψ ∝ ||x - c₀||² · ψ
- δS/δψ ∝ (1 + log p) · ψ

4.3 Connection to Imaginary-Time Schrödinger
The current evolution:

text
∂ψ/∂t = D · ∇²ψ - V(x) · ψ

is equivalent to imaginary-time Schrödinger evolution; anchor potential encodes the energy landscape.

5. Derived Quantities
5.1 Effective Temperature of Relaxation
text
T_eff = ΔU / ΔS

Guard:
- if |ΔS| < ε_S ⇒ T_eff = NaN (undefined)
- log status

5.2 Free Energy Reduction
text
ΔF = F_final - F_initial
R_F = (F_initial - F_final) / |F_initial|

Report which quantity is used (R_F or R_U).

5.3 Equilibration Time
text
τ_eq = time at which |dF/dt| < ε for k consecutive steps

6. Implementation
6.1 Per-Step Computation (GPU reduction)
cpp
struct FreeEnergyMetrics {
    double U;           // internal energy (displacement)
    double S;           // entropy (bits or nats)
    double T;           // temperature (constant or derived)
    double F;           // free energy = U - T*S
    double dF_dt;       // rate of change (from previous step)
};

__global__ void compute_free_energy(
    const double* psi_sq,       // |ψ|² per grid point
    const double* grid_coords,  // (x,y) per grid point
    const double* c0,           // baseline centroid [2]
    double T,                   // temperature
    double eps,                 // entropy epsilon
    FreeEnergyMetrics* out,     // output metrics
    int N                       // grid size
);

6.2 Settings Schema (run_settings.json equivalent)
toml
[
free_energy
]
enabled = true

[
free_energy.U
]
source = "field"                    # "field" | "embeddings"
reference = "loaded_centroid"       # c₀ from identity baseline

[
free_energy.S
]
source = "field"                    # "field" | "embeddings"
base = 2                            # 2 = bits, e = nats

[
free_energy.T
]
mode = "constant"                   # "constant" | "context" | "fluctuation"
value = 1.0                         # if mode = "constant"
fluctuation_window = 100            # if mode = "fluctuation"

[
free_energy.validation
]
monotonicity_mode = "trend"         # "trend" | "strict"
window_size = 100
trend_threshold = 0.90
log_violations = true

[
free_energy.units
]
normalize = true                    # if true, use U_norm/S_norm
U_scale = "baseline_std"            # "baseline_std" | "baseline_max" | explicit value
S_base = 2                          # 2 = bits, e = nats
S_max_mode = "grid"                 # S_max = log_base(N_grid_points)

[
free_energy.entropy
]
epsilon = 1e-15
log_epsilon_usage = true
warn_if_epsilon_fraction_gt = 0.5

[
identity.centroid
]
source = "loaded"                   # "loaded" | "computed" | "manual"
coord_frame = "pca_2d"              # must match grid coord_frame
validate_bounds = true
log_derivation_chain = true

[
free_energy.dimension
]
mode = "2d"                         # "2d" | "3d_full" | "2d_from_amplitude"

[
free_energy.T_eff
]
enabled = true
epsilon_S = 1e-10
log_regime = true

6.3 Output CSV (amended)
csv
timestep, U_raw, U_norm, S_raw, S_norm, T, F, dF_dt
0, 0.3763, 0.842, 4.521, 0.564, 1.0, 0.278, 0.0

7. Validation Criteria (Amended)
7.1 Trend Consistency
- Evaluate rolling mean trend of F(t)
- Report trend fidelity + violations

7.2 Equilibrium Detection
- |dF/dt| < ε for k steps OR
- F variance over window < threshold

7.3 Correspondence with Measurements
Expected for CAIROS May→Feb:
- U_may > U_feb (contraction toward centroid)
- S_may > S_feb (ordering, based on ECR data)
- F_may > F_feb (free energy minimization)
- T_eff > 0 (crystallization regime)

8. Semantic Claim (Corrected)
"F is an operational free‑energy analog, not a fundamental thermodynamic quantity. It combines geometric displacement from identity baseline (U) and information‑theoretic entropy (S) via a temperature‑weighted sum. The analogy to Helmholtz free energy is structural, not ontological."

9. Summary for Codex
text
F = U - T·S

U_t = Σ_x |ψ_x|² · ||x - c₀||² / Σ_x |ψ_x|²
S_t = -Σ_x p_x · log_b(p_x)
T = constant or derived
F_t = U_t - T · S_t

This is an operational analog. Log it. Plot it. Trend‑validate it.

---
Signed: Codex (Lead Engineer)
Date: 2026-02-18
