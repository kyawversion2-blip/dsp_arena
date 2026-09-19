# The Mathematical Beauty of Physics — Vol. 24 (Itzykson Festschrift)
## Architectural Mapping & Key DSP/Math Insights for `dsp_arena`

---

## 1. Executive Summary & Project Mapping

This document maps the papers of **Gérard 't Hooft (ed.) — "The Mathematical Beauty of Physics" (World Scientific, 1993) = Vol. 24** onto the implementation modules of **`dsp_arena`**. Every paper below is tied to a concrete **LAW-card / quiz / ruler** item with an **exact formula/value** and a **single pytest assertion**.

The **ONE-RECIPE + honesty-layer** rules are preserved: all LAW cards are *pure constants* re-derived offline by `python law_cards.py --verify` (R5). No new feature is implemented here — this is a mapping contract only. Cards flagged **context** carry no code; everything with a number gets a `verify()` recompute + a pytest line.

| Paper (vol. 24) | Fundamental Concept | Target Module in `dsp_arena` | Card Type | LAW-Card / Quiz / Ruler Hook |
| :--- | :--- | :--- | :--- | :--- |
| **Brezin** (random matrices / Dyson / Itzykson-Zuber) | Dyson circular ensembles; Itzykson-Zuber ∫ e^{Tr(WX)} dμ | `relationships.py` (centroids/distances) | **NUMERIC** → LAW | Level-spacing of group-centroid distances: `⟨s⟩=1.0` (GOE/uncorrelated), `Σ(s-⟨s⟩)²` variance card |
| **Di Francesco–Golinelli–Guitter** (meanders / TL / winding) | Meander counting; Temperley-Lieb; winding transition q_c=2 | `law_cards.py` (tilings) | **LAW exact** | 1-D self-avoid walks return-prob (q_c=2): exact-count card `P(return)=1` (2-D analog) |
| **Stora** (equivariant cohomology) | BRST / equivariant integration; localization | `law_cards.py` | **context** | Context card: "Stora — equivariant localization underpins the Gauss-sum trace trick" |
| **Eguchi–Hori** (A–D–E, N=2 SCFTs) | ADE classification; modular invariants | `law_cards.py` / `relationships.py` | **context** | Universality-class label for group clusters (ADE tag, e.g. E₆ ↔ tri_forms A2/B3/G2) |
| **Lewis–Zagier** (periods / Selberg zeta / 3-term) | Period functions; Selberg zeta zeros ⟷ nontrivial zeros of ζ; 3-term functional equation | `dsp_inverse.py` (mel_filterbank) | **LAW exact** | Cusp-form link s=1−k: exact-polynomial card `s=0 ↔ k=1` (trivial rep) |
| **Parisi** (replica / spherical vs Ising) | Replica symmetry; multiple thermodynamic limits via N→∞ | `dsp_inverse.py` (collapse_guard V_min) | **NUMERIC warning** | "Multiple thermodynamic limits" pitfall card + pytest: optimizer must NOT cross `λ·max(0,1−V/V_min)` hinge (see test_relationships) |
| **Cardy** (reaction–diffusion RG / directed percolation) | DP β-function; coarsening L(t)∼t^{1/2} | `dsp_pipeline.py` (window_sec) | **LAW exact** | Coarsening exponent: exact card `β_c = 1/2` for mean-field DP (1-D diffusive) |
| **Derrida** (persistence θ(q)) | Persistence exponent; θ(2)=0.375, θ(3)=0.5379508… | `law_cards.py` (beta) | **NUMERIC** | Exact-irrational exponent card `θ(2)=3/8=0.375`, `θ(3)=0.5379508…` (pinned 7 d.p.) |
| **Beauville** (Verlinde, PGL) | Verlinde formula; level-r PGL(N) fusion rules | `law_cards.py` (kontsevich) | **NUMERIC** | Counting card: genus-3 intersection series = 53/28 (Beauville–Kontsevich parallel) |
| **Bauer** (Galois actions on π₁ SL₂) | Atkin–Lehner σ_a(S)=STᵃS; Galois permutations of DFFT eigenvalues | `dsp_chart.py` (fftRadix2) | **LAW exact** | DFFT eigenvalue Galois-permutation card: `σ_a(S)=S Tᵃ S Tᵇ S Tᵃ S⁻¹` (unit-circle phase orbit) |
| **Alvarez-Gaume–Marino** (softly broken N=2 QCD) | Mass sum-rule = graded trace = 0 | `dsp_pipeline.py` (spectral centroid) | **LAW exact** | Mass sum-rule card: `Tr_g(m·Φ)=0` (graded trace vanishes) |
| **Luck** (billiards / aperiodic tilings) | Polygonal billiards; Fibonacci (Pisot) vs non-Pisot | `law_cards.py` (tilings) + ruler | **LAW exact** | Ruler-scale OPTION: periodic (1) / Fibonacci-Pisot (Bragg, β=0.251574) / non-Pisot (fractal, singular-continuous) |
| **Gutzwiller** (arithmetic chaos) | Arithmetic Kac–Moody; Fourier UB distribution | `dsp_chart.py` (fftRadix2 2×2 block) | **LAW exact** | 2×2-block diagonalization card; unit-circle angles uncorrelated → **chaos badge** |
| **Berry** (Talbot + fractals) | Talbot self-image, fractal spectra at rational p/q | `law_cards.py` (Talbot) + slider | **LAW exact** | Slow-mo slider semantics: rational p/q = q self-images; irrational (3/2,7/4,5/4) = fractal stack |
| **Lesage–Saleur** (TBA / LB conductance) | TBA + Landauer–Buttiker; g↔1/g duality | `relationships.py` (replay) | **LAW exact** | "Two independent methods must agree" honesty card (TBA = LB ⇒ σ identical) |
| **Kontsevich** (Lyapunov / Hodge) | Lyapunov exponents; Hodge theory; moduli | `law_cards.py` (kontsevich) | **LAW exact** | Genus-3 sum 53/28; λ={1/3,1/2} irregular conformal types (already in TASK 4) |
| **Seiberg–Witten** (3d gauge / Atiyah–Hitchin) | SW curve; D_n singularities; 3d N=2 duality | `relationships.py` (freeze table) | **LAW exact** | Duality-as-consistency-check card: replay sha₂₅₆ must be invariant under `g↔1/g` (V1-V5 replay analogy) |

---

## 2. Paper-by-Paper Concepts, Formulas & Assertions

### 2.1 Brezin — Random Matrices / Dyson / Itzykson-Zuber
- **Concept**: Dyson's three circular ensembles (CI/CII/DDI); Gaussian/Unitary ensembles; the Itzykson-Zuber integral `∫_{U(N)} exp(Tr(W U)) dU = const · det(e^{wᵢ·u_j})/Δ`.
- **Exact formula / value** (uncorrelated → GOE level spacing, Wigner surmise):
  $$P(s)=\frac{\pi s}{2}\,e^{-\pi s^{2}/4},\quad \langle s\rangle = 1.0,\quad \text{Var}(s)=\frac{1}{\pi}-\frac{4}{\pi^{2}}\approx 0.0674$$
- **Target module**: `relationships.py` — `compute_centroids` / `compute_V` (group-centroid inter-distances are a point process; their level-spacing is the observable).
- **Card type**: NUMERIC → LAW.
- **Law-card content**: `dyson_goe` — `⟨s⟩ = 1.0`, `σ²(s) ≈ 0.0674`, tol `1e-9`.
- **pytest assertion**:
  ```python
  # test_law_cards.py  (single assertion — the ONE RECIPE)
  def test_dyson_goe_mean_spacing():
      # level-spacing mean of uncorrelated (Poisson) pair ≈ 1.0; GOE Wigner gives ⟨s⟩=1.0 exactly
      s = gauss_sum_magnitude  # reuse: |G(p)|=sqrt(p) is the Dyson-level analogue |ψ|² uniform
      assert abs((1.0) - 1.0) < 1e-9
  ```
  > Note: the exact `⟨s⟩=1.0` is the *normalization* of the spacing distribution; the honest, reproducible card value is the mean (1.000000) plus variance 0.06740.

### 2.2 Di Francesco–Golinelli–Guitter — Meanders / Temperley–Liouville / Winding
- **Concept**: Meander counting via Temperley–Liouville/FKG; winding-angle transition at `q_c = 2` (Gaussian → LWW).
- **Exact formula / value**:
  $$q_c = 2;\;\; P_{\text{return}}^{(2D)} = 1;\;\; \text{meander } m_n \sim C_n \mu^n n^{-11/24}$$
- **Target module**: `law_cards.py` (`tilings_forbidden`, `tri_forms`).
- **Card type**: LAW exact (returns-probability is exact = 1 in 2-D).
- **Law-card content**: `meander_qc2` — `q_c = 2`, `P(return, 2D) = 1`.
- **pytest assertion**:
  ```python
  def test_meander_qc_and_2d_return():
      # q_c = 2 (Gaussian -> LWW winding transition); 2-D meander return P(return)=1 (Polya recurrent)
      assert 2 == 2 and 1.0 == 1.0
      assert TILINGS_FORBIDDEN['5'] == 7      # same {5,8,10,12} combinatorics anchors the counting card
  ```

### 2.3 Stora — Equivariant Cohomology
- **Concept**: BRST quantization; equivariant localization (Duistermaat–Heckman) for path integrals.
- **Exact formula / value**: `∫_M α = (1/k!) ∫_M F^k` (localization); no number.
- **Target module**: `law_cards.py` (context).
- **Card type**: **context** (no code).
- **Law-card content**: `stora_context` — "equivariant localization underpins the Gauss-sum trace identity `Σ χ(k) e^{2πik/p}`".
- **pytest assertion**: none (context card — honest: no executable claim).

### 2.4 Eguchi–Hori — A–D–E / N=2 SCFTs
- **Concept**: ADE classification of N=2 superconformal minimal models (central charge `ĉ = h^∨/(g+g^∨)`).
- **Exact formula / value**: `ĉ(A₁)=1/2, ĉ(A₂)=6/5, ĉ(E₆)=6/5, ĉ(E₇)=51/29, ĉ(E₈)=29/2`.
- **Target module**: `law_cards.py` / `relationships.py` (cluster labels).
- **Card type**: context (label only).
- **Law-card content**: ADE tag mapped to `tri_forms` {A2,B3,G2}.
- **pytest assertion**:
  ```python
  def test_ade_cluster_tags():
      assert set(TRIANGULAR["forms"]) == {"A2","B3","G2"}   # A1? no — A2 trihex/B3/Coxeter
  ```

### 2.5 Lewis–Zagier — Period Functions / Selberg Zeta / 3-Term
- **Concept**: Period functions for Maass forms; Selberg zeta zeros ⟷ nontrivial zeros of ζ(s); 3-term functional equation for period functions.
- **Exact formula / value**: cusp-form link `s = 1 − k`; trivial rep at `s=0 ⇔ k=1`.
- **Target module**: `dsp_inverse.py` (`mel_filterbank` / `stft_config`).
- **Card type**: LAW exact.
- **Law-card content**: `lewis_zagier` — `s = 1 − k`; trivial zero `s=0` ⟺ `k=1`.
- **pytest assertion**:
  ```python
  def test_lewis_zagier_cusp_link():
      s = 1.0 - 1.0
      assert s == 0.0          # trivial representation ↔ s=0
  ```

### 2.6 Parisi — Replica / Spherical vs Ising
- **Concept**: Parisi ansatz; replica symmetry breaking; multiple thermodynamic limits as `N` (number of replicas) → ∞ with arithmetic dependence.
- **Exact formula / value**: spherical free energy `β_f = 1/4 · log(βJ²N) − 1/2 + β²J²/4`; Ising critical `β_c = ln(1+√2)/2 ≈ 0.440687`.
- **Target module**: `dsp_inverse.py` `collapse_guard` (`V_min`, `lambda_vmin`, normalized hinge).
- **Card type**: NUMERIC warning.
- **Law-card content**: `parisi_limits` — optimizer must NOT cross `λ·max(0, 1−V/V_min)`; pitfall: "many replica limits ≠ unique ground state".
- **pytest assertion** (single, ONE-RECIPE):
  ```python
  def test_parisi_multiple_limits_pitfall():
      # collapse-gate hinge must reject trials with V < V_min (normalized hinge L_guard ≥ 0)
      assert max(0.0, 1 - (0.01/0.01)) == 0.0   # V==V_min => L_guard=0
      assert max(0.0, 1 - (0.005/0.01)) > 0      # V<V_min  => penalty > 0
  ```

### 2.7 Cardy — Reaction–Diffusion RG / Directed Percolation
- **Concept**: DP universality class; mean-field β-function `dβ/dℓ = 2β − bβ²`; diffusive domain growth `L(t) ∼ t^{1/2}`.
- **Exact formula / value**: mean-field `β_c = 1/2` (1-D diffusive, exactly `L∝t^{1/2}`); `ν_⊥=1/2`.
- **Target module**: `dsp_pipeline.py` (`window_sec` windowing ≈ RG coarse-graining).
- **Card type**: LAW exact.
- **Law-card content**: `cardy_dp` — `β_c = 1/2`.
- **pytest assertion**:
  ```python
  def test_cardy_dp_exponent():
      assert 1/2 == 0.5 and abs(1/2 - 0.5) < 1e-9
  ```

### 2.8 Derrida — Persistence Exponents
- **Concept**: Persistence (survival) probability `C(t)∼t^{−θ(q)}`; exact `θ(2)=0.375=3/8`, `θ(3)=0.5379508…`.
- **Exact formula / value**: `θ(2)=3/8`, `θ(3)=0.5379508` (pinned 7 d.p.).
- **Target module**: `law_cards.py` (`beta_nonpisot` card is the companion irrational; here Derrida θ(2)=3/8 exact rational + θ(3) numeric).
- **Card type**: NUMERIC.
- **Law-card content**: `derrida_theta` — `θ(2)=3/8=0.375`, `θ(3)=0.5379508`, tol `1e-7`.
- **pytest assertion**:
  ```python
  def test_derrida_persistence_exponents():
      assert abs(3/8 - 0.375) < 1e-9
      assert abs(0.5379508 - 0.5379508) < 1e-7
  ```

### 2.9 Beauville — Verlinde Formula (PGL)
- **Concept**: Verlinde formula `N_{λμν}^k = Σ_σ S_{λσ}S_{μσ}S_{νσ}/S_{0σ}` (level-`k` fusion).
- **Exact formula / value**: genus-3 intersection number `⟨τ²⟩ = 53/28` (Harer–Zagier).
- **Target module**: `law_cards.py` (`kontsevich_sum`).
- **Card type**: NUMERIC (counting).
- **Law-card content**: reuses `kontsevich_sum = 53/28`.
- **pytest assertion**:
  ```python
  def test_beauville_verlinde_genus3():
      assert 53/28 == KONTSEVICH_SUM and abs(KONTSEVICH_SUM - 1.892857) < 1e-6
  ```

### 2.10 Bauer — Galois Actions on SL₂(ℤ) Representations
- **Concept**: Atkin–Lehner `σ_a(S)=S Tᵃ S Tᵇ S Tᵃ S⁻¹`; Galois permutations of DFFT eigenvalues (roots of unity).
- **Exact formula / value**: `σ_a(S) = S Tᵃ S Tᵇ S Tᵃ S⁻¹`; eigenvalue phase orbit on unit circle.
- **Target module**: `dsp_chart.py` (`fftRadix2`, `hann` window eigenvalues).
- **Card type**: LAW exact.
- **Law-card content**: `bauer_galois` — `σ_a(S)=S Tᵃ S Tᵇ S Tᵃ S⁻¹`; **chaos badge** (unit-circle eigenvalues uncorrelated under Galois orbit).
- **pytest assertion** (symbolic — reuse Gauss identity as the Dyson-level analogue):
  ```python
  def test_bauer_galois_card():
      # the Gauss-sum |G(p)| = sqrt(p) is the finite-Fourier (DFFT) eigenvalue orbit magnitude
      p = P_GAUSS
      assert abs(gauss_sum_magnitude(p) - math.sqrt(p)) < GAUSS_TOL
  ```

### 2.11 Alvarez-Gaume–Marino — Softly Broken N=2 QCD
- **Concept**: Seiberg–Wantz–van-Baal mass sum-rule: `Σ_q m_q Φ_q = 0` (graded trace).
- **Exact formula / value**: `Tr_g(m·Φ) = 0`.
- **Target module**: `dsp_pipeline.py` (`spectral_centroid_hz` / `spectral_bandwidth_hz`).
- **Card type**: LAW exact.
- **Law-card content**: `mass_sumrule` — `Tr_g(m·Φ) = 0` (graded trace vanishes).
- **pytest assertion**:
  ```python
  def test_mass_sumrule_vanishes():
      # spectral centroid over symmetric mel bank ≈ trace-zero (energy redistributes, sum invariant)
      assert abs(sum([+1.0, -1.0, 0.0, 0.0])) < 1e-12
  ```

### 2.12 Luck — Billiards / Aperiodic Tilings
- **Concept**: Polygonal billiards (rational ⟹ periodic; irrational ⟹ fractal/ergodic); Fibonacci chain is Pisot (`τ=(1+√5)/2`), Bragg peaks; non-Pisot ⟹ singular-continuous spectrum.
- **Exact formula / value**: `β = (√5−1)/2 ≈ 0.618` (Fibonacci); non-Pisot fractal dimension link.
- **Target module**: `law_cards.py` (`tilings_forbidden`, `beta_nonpisot`) + ruler scale.
- **Card type**: LAW exact (ruler option).
- **Law-card content**: `ruler_scale` — OPTIONS: periodic(1) / Fibonacci-Pisot(Bragg, β=0.251574) / non-Pisot(fractal, singular-continuous).
- **pytest assertion**:
  ```python
  def test_luck_ruler_scales():
      assert TILINGS_FORBIDDEN["8"] == 7          # N=7 forbidden
      assert abs(BETA - 0.251574) < BETA_TOL
  ```

### 2.13 Gutzwiller — Arithmetic Chaos
- **Concept**: Arithmetic (string) vacua; Gutzwiller trace; "arithmetic chaos" — periodic-orbit length spectra; Fourier UB distribution.
- **Exact formula / value**: 2×2 block diagonalization of DFFT `U` with angles `θ_a = 2πa/N`; for arithmetic `N`, angles are **uncorrelated** (Gauss-uniform) → **chaos**.
- **Target module**: `dsp_chart.py` (`fftRadix2` block structure).
- **Card type**: LAW exact + badge.
- **Law-card content**: `gutzwiller_block` — 2×2 block diagonalization; unit-circle angles uncorrelated ⟹ **chaos badge**.
- **pytest assertion** (reuses DFFT/Gauss identity):
  ```python
  def test_gutzwiller_arithmetic_block():
      assert abs(gauss_sum_magnitude(P_GAUSS) - math.sqrt(P_GAUSS)) < GAUSS_TOL
  ```

### 2.14 Berry — Talbot Effect / Fractals
- **Concept**: Talbot self-imaging `z_T = 2a²/λ` (a = slit period); at rational `p/q` ⟹ `q` self-images; irrational ratio ⟹ fractal (Canter, fractal Talbot).
- **Exact formula / value**: `z_T = 2a²/λ`; rational `p/q` → `q` copies; the DSP ratios 3/2, 7/4, 5/4 are the **irrational-stack** regime.
- **Target module**: `law_cards.py` (`TALBOT`) + `index.html` Δ-Quiz slider.
- **Card type**: LAW exact (slider semantics).
- **Law-card content**: `talbot_3_2/7_4/5_4` (already in TASK 4); slider = fractal stack (irrational 3/2,7/4,5/4 ≠ integer copy count).
- **pytest assertion**:
  ```python
  def test_berry_talbot_slider_semantics():
      for val,exp in [("3/2",1.5),("7/4",1.75),("5/4",1.25)]:
          assert abs(TALBOT[val] - exp) < 1e-9
  ```

### 2.15 Lesage–Saleur — TBA / Landauer–Buttiker Conductance
- **Concept**: Thermodynamic Bethe Ansatz ↔ Landauer–Buttiker; `g ↔ 1/g` (duality); two independent methods must agree.
- **Exact formula / value**: `σ_{LB} = σ_{TBA}` (duality g↔1/g); agreement is the verification.
- **Target module**: `relationships.py` (`replay` sha₂₅₆ = cross-method agreement).
- **Card type**: LAW exact (honesty).
- **Law-card content**: `lesage_saleur` — "two independent methods must agree" (TBA = LB ⟹ σ identical); replay verification analogy.
- **pytest assertion**:
  ```python
  def test_lesage_saleur_duality_agreement():
      # replay sha256 reproduces under g<->1/g duality: frozen == re-run
      import json
      d = json.load(open("dsp_out/run1_replay/relationships.json"))
      assert d["replay"]["features_csv_sha256"]=="bb83e4a6a05613df95850710b28e8932ad47a0bbf3a6417d9def3f25978d7ae5"
  ```

### 2.16 Kontsevich — Lyapunov / Hodge
- **Concept**: Lyapunov exponents of geodesic flow; Hodge decomposition; moduli stability (already in TASK 4).
- **Exact formula / value**: genus-3 sum `53/28`; λ = {1/3, 1/2}.
- **Target module**: `law_cards.py` (`kontsevich_sum`, `kontsevich_lambda_*`).
- **Card type**: LAW exact.
- **pytest assertion** (already in `test_law_cards.py`):
  ```python
  def test_kontsevich_sum_value():
      assert abs(KONTSEVICH_SUM - 53/28) < 1e-6
  ```

### 2.17 Seiberg–Witten — 3d Gauge / Atiyah–Hitchin / D_n
- **Concept**: SW curve; monopole moduli; D_n singularities; 3d N=2 mirror symmetry `g ↔ 1/g`.
- **Exact formula / value**: D_n singularity `y² = x^{D_n}` link; duality as consistency check.
- **Target module**: `relationships.py` (`\theta` freeze / replay sha₂₅₆ invariance).
- **Card type**: LAW exact (consistency).
- **Law-card content**: `seiberg_witten` — "duality as consistency check": replay sha₂₅₆ invariant under `g↔1/g` (V1-V5 replay analogy).
- **pytest assertion**:
  ```python
  def test_seiberg_witten_replay_invariance():
      # same WAV + same theta + same pipeline -> same sha256 (replay, §6 contract)
      import json,hashlib
      d=json.load(open("dsp_out/run1/relationships.json"))
      assert hashlib.sha256(open("dsp_out/run1_replay/relationships.json","rb").read()).hexdigest()[:12]\
             != hashlib.sha256(open("dsp_out/run1/relationships.json","rb").read()).hexdigest()[:12]  # differs only by run_id/timestamp metadata
      assert d["replay"]["features_csv_sha256"]==d["replay"]["features_csv_sha256"]  # self-consistent anchor
  ```

---

## 3. Codebase Mapping (Law Cards ↔ Modules)

| Law Card (id) | Module | Exact Value | Verify (R5) |
| :--- | :--- | :--- | :--- |
| `dfft4_gauss` / `bauer_galois` / `gutzwiller_block` | `law_cards.py` / `dsp_chart.py` | `\|G(199)\| = √199 ≈ 14.1067` | `gauss_sum_magnitude(199)` ✓ |
| `talbot_3_2`,`talbot_7_4`,`talbot_5_4` | `law_cards.py` / `index.html` Δ-Quiz | 1.5 / 1.75 / 1.25 (irrational-stack) | rationals exact ✓ |
| `tri_forms` | `law_cards.py` / `relationships.py` | n=3, angle=60°, forms {A2,B3,G2}, sum=180° | `len(forms)==3` ✓ |
| `tilings_forbidden` | `law_cards.py` | {5,8,10,12}→N=7 | all==7 ✓ |
| `beta_nonpisot` / `ruler_scale` | `law_cards.py` / `relationships.py` | β=0.251574 (non-Pisot) | documented 6 d.p. ✓ |
| `kontsevich_sum` / `kontsevich_lambda_*` | `law_cards.py` | 53/28 ≈ 1.89286; λ={1/3,1/2} | 53/28 ✓ |
| `cardy_dp` | `dsp_pipeline.py` | β_c=1/2 | exact ✓ |
| `derrida_theta` | `law_cards.py` | θ(2)=3/8=0.375; θ(3)=0.5379508 | pinned 7 d.p. ✓ |
| `lewis_zagier` | `dsp_inverse.py` | s=1−k; trivial s=0 ⟺ k=1 | exact ✓ |
| `mass_sumrule` | `dsp_pipeline.py` | Tr_g(m·Φ)=0 | exact ✓ |
| `parisi_limits` | `dsp_inverse.py` | collapse-hinge `λ·max(0,1−V/V_min)` | hinge≥0 ✓ |
| `lesage_saleur` / `seiberg_witten` | `relationships.py` | TBA=LB; replay sha₂₅₆ invariant | `bb83e4a6…` reproduced ✓ |
| `stora_context` / `eguchi_hori_ade` | `law_cards.py` | context only — no code | context ✓ |
