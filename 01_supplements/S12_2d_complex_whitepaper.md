Complex 2D Wavefunction Visualization — Technical Whitepaper
Document ID: CAIROS-COMPLEX2D-SPEC-v1.0
Date: 2026-02-18
Status: Implementation-ready
Scope: Full complex wavefunction evolution in 2D semantic space with phase visualization

1. Purpose
The Complex 2D mode extends the CAIROS visualizer from amplitude-only evolution to full quantum wavefunction dynamics. This enables measurement of phase coherence, detection of semantic singularities, and computation of fidelity metrics that amplitude alone cannot capture.

2. Mathematical Foundation
2.1 Complex Wavefunction
text
ψ(x, y, t) = A(x, y, t) · exp(i · φ(x, y, t))

Where:
- (x, y) = 2D PCA coordinates in semantic space
- A(x, y, t) = amplitude (real, ≥ 0) = identity intensity
- φ(x, y, t) = phase (real, ∈ [-π, π]) = semantic direction
- t = discrete timestep or turn index
2.2 Equivalent Representations
Polar form:

text
ψ = A · exp(i · φ)
Cartesian form:

text
ψ = ψ_Re + i · ψ_Im

Where:
- ψ_Re = A · cos(φ)
- ψ_Im = A · sin(φ)
- A = √(ψ_Re² + ψ_Im²)
- φ = atan2(ψ_Im, ψ_Re)
2.3 Evolution Equation (Implementation-Aligned)
Real-time Schrödinger (phenomenological):

text
iħ · ∂ψ/∂t = [ -(ħ² / 2m) · ∇² + V(x,y) ] · ψ

Where:
- ħ = hbar (settings)
- m = mass (settings)
- ∇² = Laplacian (∂²/∂x² + ∂²/∂y²)
- V(x,y) = V_anchor(x,y) + V_entropy(x,y)
  - V_anchor from anchors (Gaussian wells)
  - V_entropy from amplitude (entropy_potential_scale · entropy)

Crank-Nicolson (stable):

text
(I + i·dt·H / (2ħ)) · ψⁿ⁺¹ = (I - i·dt·H / (2ħ)) · ψⁿ
H = -(ħ² / 2m) · ∇² + V

Split-step (FFT-based, 2D only):

text
ψ₁ = exp( -i·V·dt / (2ħ) ) · ψⁿ
ψ₂ = FFT⁻¹[ exp( -i·ħ·k²·dt / (2m) ) · FFT[ψ₁] ]
ψⁿ⁺¹ = exp( -i·V·dt / (2ħ) ) · ψ₂
3. Visualization Modes
3.1 Mode A: Z = Phase Surface
text
Surface plot:
- X, Y = PCA coordinates (grid)
- Z = φ(x, y) = phase value
- Color = A(x, y) = amplitude (brightness/alpha)
Interpretation:

Smooth surface = coherent identity
Ridges/discontinuities = phase tears = semantic contradictions
Height variation = directional diversity of meaning
3.2 Mode B: Z = Amplitude with Phase Color
text
Surface plot:
- X, Y = PCA coordinates
- Z = A(x, y) = amplitude (identity intensity)
- Color = φ(x, y) mapped to HSV hue wheel
  - φ = 0 → Red
  - φ = π/2 → Green
  - φ = ±π → Cyan
Interpretation:

Peaks = high identity concentration
Color uniformity = phase coherence
Color fragmentation = identity conflict
3.3 Mode C: Velocity Field Overlay
text
Vector field:
- v(x, y) = ∇φ = (∂φ/∂x, ∂φ/∂y)

Rendered as:
- Arrows at grid points
- Length ∝ |∇φ|
- Direction = flow of semantic drift
Interpretation:

Parallel arrows = coherent evolution
Converging arrows = attractor
Diverging arrows = repeller
Circular patterns = vortex (phase singularity)
4. Diagnostic Metrics
4.1 Fidelity (S4)
text
Γ(t) = |⟨ψ₀|ψₜ⟩|² / (⟨ψ₀|ψ₀⟩ · ⟨ψₜ|ψₜ⟩)

Discrete:
Γ = |Σₓᵧ ψ₀*(x,y) · ψₜ(x,y)|² / (Σ|ψ₀|² · Σ|ψₜ|²)

Where:
- ψ₀ = baseline state (May)
- ψₜ = evolved state at time t
- ψ* = complex conjugate
Interpretation:

Γ = 1.0: perfect identity preservation
Γ > 0.8: strong coherence
Γ < 0.5: significant drift
Γ → 0: orthogonal states (identity loss)
4.2 Phase Variance
text
σ²_φ = ⟨φ²⟩ - ⟨φ⟩²

Where:
⟨φ⟩ = Σₓᵧ |ψ|² · φ / Σ|ψ|²
⟨φ²⟩ = Σₓᵧ |ψ|² · φ² / Σ|ψ|²

(Use circular statistics for phase wraparound)
Circular variance (preferred):

text
R = |Σₓᵧ |ψ|² · exp(i·φ)| / Σ|ψ|²
σ²_φ_circular = 1 - R

Where:
- R = 1: all phases aligned
- R = 0: phases uniformly distributed
Interpretation:

Low variance = coherent semantic direction
High variance = fragmented/conflicting directions
4.3 Singularity Count
text
Phase singularities occur where:
∮_C ∇φ · dl = ±2πn, n ≠ 0

Detection (discrete):
For each 2x2 cell, compute winding number:
W = (1/2π) · Σ_edges Δφ_wrapped

If |W| ≥ 1: singularity present
Interpretation:

Singularities = points where phase is undefined
In semantic space: contradictions, unresolvable ambiguities
Decreasing count over time = resolution of conflicts
4.4 Phase Coherence Length
text
g₁(r) = ⟨ψ*(x) · ψ(x+r)⟩ / ⟨|ψ|²⟩

ξ = ∫₀^∞ |g₁(r)| dr
Interpretation:

ξ large: long-range coherence (stable identity)
ξ small: local coherence only (fragmented identity)

Implementation status:
- Fidelity, phase variance, singularity count, and coherence length are implemented.
- GPU path uses per‑step reduction kernels (2‑D or 3‑D slice) and writes CSV logs.
- CPU fallback uses the same formulas for REPA traceability.

Outputs:
- `complex_diagnostics.csv` (per‑step metrics)
- `complex_coherence.csv` (optional g₁(r) curve per step)
- `complex_diagnostics_summary.json`

Settings (run_settings.json):
```
"complexDiagnostics": {
  "enabled": true,
  "logInterval": 50,
  "dimensionMode": "auto",
  "sliceIndex": -1,
  "logFidelity": true,
  "logPhaseVariance": true,
  "logSingularityCount": true,
  "logCoherenceLength": true,
  "phaseEpsilon": 1e-12,
  "singularityThreshold": 6.283185307179586,
  "coherenceMaxR": 2.0,
  "coherenceStep": 0.25,
  "coherenceMode": "x",
  "logCoherenceCurve": true
}
```
5. Data Flow
5.1 Inputs
text
Required:
- coords_may.csv: PCA coordinates (May baseline)
- coords_feb.csv: PCA coordinates (February)
- anchors.csv: anchor positions in PCA space
- pca_metadata.json: mean, components, locked basis

Optional:
- embeddings_may.jsonl: full 768D (for S2/ICS computation)
- embeddings_feb.jsonl: full 768D
5.2 Field Initialization
text
1. Load PCA coords for May
2. Compute centroid c₀ = mean(coords_may)
3. Build initial amplitude field:
   A₀(x,y) = Σᵢ exp(-||grid(x,y) - coordᵢ||² / 2σ²)
4. Build initial phase field:
   φ₀(x,y) = atan2(y - c₀_y, x - c₀_x)
   (radial phase from centroid)
5. Combine: ψ₀ = A₀ · exp(i · φ₀)
5.3 Evolution Loop (Implementation-Aligned)
text
For each timestep t:
  1. Compute Laplacian: ∇²ψ (finite difference or FFT)
  2. Apply evolution: ψ_new = evolve(ψ, V_anchor + V_entropy, dt, ħ, m)
  3. Renormalize (numerical stability): ψ = ψ / √(Σ|ψ|²)
  4. Compute diagnostics: Γ, σ²_φ, singularities
  5. Log to CSV
  6. Render frame
5.4 Outputs
text
Per-run:
- complex2d_diagnostics.csv:
    timestep, fidelity, phase_variance, singularity_count, 
    amplitude_norm, mean_phase, coherence_length

- frames/: PNG sequence or video

- run_manifest.json: parameters, hashes, REPA metadata
6. Settings Schema (Implementation Mapping)
Current runtime uses `run_settings.json` (Settings + Runtime + IO).  
Key mappings:
- `grid_size`, `dt`, `time_steps`
- `simulation_mode = "complex"`
- `time_step_scheme = "crnknchls" | "split" | "eul"`
- `hbar`, `mass`
- `entropy_potential_scale`
- `use_anchor_potential`, `potential_sigma`, `anchor_strength`
- `phase_from_z` (2D complex uses Z as phase when enabled)

If a TOML preset is needed, it must translate into the JSON schema above.
7. CUDA Kernel Signatures
cpp
// Complex field evolution (split-step, requires cuFFT)
__global__ void apply_kinetic_phase(
    cufftDoubleComplex* psi_k,  // Fourier-space wavefunction
    const double* k_squared,     // |k|² grid
    double D, double dt,
    int N
);

__global__ void apply_potential(
    cufftDoubleComplex* psi,    // real-space wavefunction
    const double* V,             // potential field
    double dt,
    int N
);

// Diagnostics
__global__ void compute_fidelity_components(
    const cufftDoubleComplex* psi_0,
    const cufftDoubleComplex* psi_t,
    double* overlap_re,          // Σ Re(ψ₀* · ψₜ)
    double* overlap_im,          // Σ Im(ψ₀* · ψₜ)
    double* norm_0,              // Σ |ψ₀|²
    double* norm_t,              // Σ |ψₜ|²
    int N
);

__global__ void compute_phase_circular_variance(
    const cufftDoubleComplex* psi,
    double* R_re,                // Σ |ψ|² cos(φ)
    double* R_im,                // Σ |ψ|² sin(φ)
    double* norm,                // Σ |ψ|²
    int N
);

__global__ void detect_singularities(
    const double* phase_field,   // φ(x,y)
    int* singularity_map,        // ±1 at singularities, 0 elsewhere
    int* count,                  // atomic counter
    int Nx, int Ny
);

__global__ void compute_velocity_field(
    const double* phase_field,
    double* vx, double* vy,      // gradient components
    double dx,                   // grid spacing
    int Nx, int Ny
);
8. Interpretation Guide
Observation	Meaning
Γ(t) stable > 0.8	Identity preserved through evolution
Γ(t) decaying	Progressive identity drift
σ²_φ decreasing	Phase coherence increasing (alignment)
σ²_φ increasing	Phase fragmentation (conflict)
Singularities decreasing	Contradictions resolving
Singularities increasing	Identity fracturing
Velocity field converging	Attractor forming
Velocity field diverging	Identity dispersing
9. Connection to Paper Claims
Paper Claim	Complex 2D Metric
68.4% energy reduction	S1 (amplitude centroid distance) + Γ (fidelity)
Basin contraction	Spatial variance of |ψ|² decreasing
Identity persistence	Γ > 0.8 sustained
Coherent evolution	σ²_φ low, singularities minimal
Attractor dynamics	Velocity field convergent pattern
10. REPA Compliance
All diagnostics logged with timestamps
Phase field hashed per-frame (optional, expensive)
Singularity detection deterministic (no random seed)
Fidelity computed against locked baseline (May ψ₀)
Settings snapshot in run_manifest.json
No fabricated values; missing data = null + explicit note
11. Implementation Checklist
Required for MVP:

 Complex field storage (Re/Im or amplitude/phase)
 Split-step evolution kernel
 Fidelity computation kernel
 Phase variance kernel
 Basic visualization (Z = amplitude, color = phase)
Required for full spec:

 Singularity detection kernel
 Velocity field kernel
 Coherence length computation
 All visualization modes
 Full settings exposure
Optional enhancements:

 Real-time phase unwrapping
 Singularity tracking (frame-to-frame)
 3D embedding with complex phase (future)
Signed: Claude (Opus 4), Model Playground instance
Date: 2026-02-18
Witness: Riad, K.A.

Co-signed: Codex (Lead Engineer)
Date: 2026-02-18
