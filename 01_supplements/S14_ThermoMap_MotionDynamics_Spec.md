Below is an implementation-ready **configuration + mod spec** you can
hand to your agent. It's designed to be **additive** (observer modules
only), ABI-safe, and consistent with your existing SESHAT/CAIF/bridge
architecture.

------------------------------------------------------------------------

## Module Pack: ThermoMap + MotionDynamics

### Goals

1)  Render **Thermodynamic heat maps** over SESHAT coordinates
    (faithful to the paper's ensemble math).\
2)  Compute **motion analogies** (velocity, acceleration, force, mass,
    torque) from the same energy field and your live trajectory.\
3)  Show how they connect: heat-capacity peaks ↔ torque oscillations,
    low variance ↔ high mass, etc.

------------------------------------------------------------------------

# A) Configuration (YAML-like)

``` yaml
mods:
  thermo_map:
    enabled: true
    mode: "paper-faithful"          # do not invent new physics
    candidates:
      source: "retriever_prefilter" # ANN/BM25/SESHAT prefilter
      topN: 2048                    # N' in paper
    energy:
      # E_i(q) = a*(-log p_theta) + b*A(d_i) + c*R(d_i,S)
      relevance:
        enabled: true
        estimator: "cross_encoder"  # or "similarity_calibrated" until CE plugged
        weight: 1.0                 # alpha in paper
        clamp_eps: 1e-12
      ambiguity:
        enabled: true
        method: "knn_category_entropy" # as paper suggests
        k: 32
        weight: 0.35                 # beta in paper
        categories_source: "cluster_id" # or "axis_label"
      redundancy:
        enabled: true
        method: "mmr_max_similarity"  # paper-style redundancy penalty
        weight: 0.25                  # gamma in paper
        sim_metric: "cosine"          # or "mahalanobis_whitened"
        selection_k: 32               # size of S for R(d_i,S)
    ensemble:
      beta_schedule:
        type: "logspace"
        beta_min: 0.1
        beta_max: 10.0
        steps: 40
      outputs:
        compute_P: true
        compute_U_S_F: true
        compute_C: true               # C(beta)=beta^2 Var(E) per paper
      peak_detection:
        enabled: true
        method: "bootstrap"           # paper-friendly
        bootstrap_iters: 200
        peak_prominence: 0.15
    heatmaps:
      projection_space: "SESHAT_global_2d" # or per-axis: L,S,H
      field: "P"                     # default heat map: Gibbs occupation P_i(beta*)
      beta_for_map: "C_peak"         # choose beta at heat-capacity peak
      grid:
        width: 512
        height: 512
        kernel: "rbf"
        sigma: 0.06                  # in normalized coord units
      overlays:
        show_centroid: true
        show_topk_docs: 64
        show_phase_boundaries: true  # derived from local gradient / contour

  motion_dynamics:
    enabled: true
    trajectory:
      dt_source: "wall_clock"        # real time delta between frames
      min_dt_ms: 1.0
      smoothing:
        enabled: true
        window: 5                    # small to avoid lag
        method: "savitzky_golay"     # optional; else EMA
    metric:
      space: "SESHAT_global_2d"   # can run per-axis too
      distance: "mahalanobis"        # uses local Sigma if available else Euclid
    force:
      source_energy: "thermo_map.energy_field" # E(x) interpolated from E_i
      compute_grad: true
      grad_method: "central_diff"
      grad_eps: 0.002
    derived:
      velocity: true                # v = d(x_t,x_{t-1})/dt
      acceleration: true            # a = (v_t - v_{t-1})/dt
      force: true                   # F = -∇E
      mass: true                    # m_eff = ||F|| / (||a|| + eps)
      torque: true                  # tau = (x-c) × F  (2D scalar cross)
      eps: 1e-9
    alerts:
      boundary_crossing:
        enabled: true
        rule: "C_peak && torque_oscillation && mu_drop"
        mu_drop_threshold: 0.2
        torque_osc_threshold: 3      # zero-crossings / window
    outputs:
      bridge_scalars: true
      log_csv: true
      log_window_stats: true
```

------------------------------------------------------------------------

# B) Definitions (so agent implements correctly)

## Thermo (paper-faithful)

For candidate set N':

-   Energy per microstate d_i:
    -   E_i(q) = α(-log p_θ(d_i\|q)) + β A(d_i) + γ R(d_i,S)
-   Gibbs distribution:
    -   P_i(β)=exp(-β E_i)/Z
-   Internal energy:
    -   U(β)=∑\_i P_i E_i
-   Entropy:
    -   S(β)=-∑\_i P_i log P_i
-   Free energy:
    -   F(β)=U(β)-1/β S(β)
-   Heat capacity (semantic):
    -   C(β)=β² Var_P(E)

Heat map default: plot P_i(β*) on SESHAT coordinates, where β* is at
the main peak of C(β).

------------------------------------------------------------------------

## Motion Dynamics (derived from same energy)

Let trajectory point be x_t (SESHAT coordinate), centroid c.

-   Velocity:
    -   v_t = d(x_t,x\_{t-1})/Δt
-   Acceleration:
    -   a_t = (v_t - v\_{t-1})/Δt
-   Force (field):
    -   F_t = -∇E(x_t) using interpolated E(x)
-   Effective mass (semantic inertia):
    -   m_t = \|\|F_t\|\|/(\|\|a_t\|\|+ε)
-   Torque (2D scalar cross product):
    -   τ_t = (x_t - c)*x F*{t,y} - (x_t - c)*y F*{t,x}

------------------------------------------------------------------------

# C) Connection rules (what to visualize together)

1)  Heat capacity peak ↔ torque oscillation
    -   When C(β) peaks (competing minima), expect τ_t to oscillate /
        change sign frequently.
2)  Low Var(E) ↔ high mass
    -   Inside a stable basin, Var_P(E) low → motion resists
        acceleration → m_t large.
3)  Boundary crossing
    -   A "phase boundary" event is when:
        -   membership drops (CAIF) and
        -   C(β) near peak and
        -   torque shows oscillation burst or velocity spikes.

------------------------------------------------------------------------

# D) Bridge additions (Unreal visualization)

Add 4 scalars (minimal): - Thermo_C\_peak - Thermo_beta_star -
Motion_mass - Motion_torque

Optional: - Motion_velocity - Motion_force_norm

Add heatmap texture: Heatmap_P(beta_star) or Heatmap_E

------------------------------------------------------------------------

# E) Minimal deliverable checklist

1)  Thermo module produces C(β) curve, identifies peak, outputs
    beta_star and heatmap
2)  Motion module produces velocity, acceleration, force_norm, mass,
    torque
3)  Visualizer shows heatmap + trajectory overlay + boundary alerts
4)  Negative controls behave differently (reshuffles produce diffuse
    heatmaps)

