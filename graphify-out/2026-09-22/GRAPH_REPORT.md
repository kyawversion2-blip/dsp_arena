# Graph Report - dsp_arena  (2026-08-20)

## Corpus Check
- 42 files · ~46,928 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 277 nodes · 425 edges · 18 communities (16 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- test_relationships.py
- dsp_inverse.py
- 2. Paper-by-Paper Concepts, Formulas & Assertions
- dsp_chart.py
- DSP Arena — TV-Game Verdict Screen (Design Spec)
- law_cards.py
- test_law_cards.py
- DSP Inverse Optimizer - Report
- DSP Inverse Optimizer - Report
- DSP Inverse Optimizer - Report
- DSP Arena — Project Memory
- DSP Arena — Interactive Acoustic Event Chart
- ref_dsp.py
- DSP Inverse Optimizer - Report
- DSP Inverse Optimizer - Report
- ._score

## God Nodes (most connected - your core abstractions)
1. `2. Paper-by-Paper Concepts, Formulas & Assertions` - 18 edges
2. `compute_centroids()` - 14 edges
3. `build_baseline_table()` - 14 edges
4. `cmd_optimize()` - 13 edges
5. `metrics_for()` - 11 edges
6. `cmd_relationships_baseline()` - 10 edges
7. `compute_distance_matrix()` - 10 edges
8. `L_guard_value()` - 10 edges
9. `DSP Arena — Project Memory` - 10 edges
10. `DSP Arena — TV-Game Verdict Screen (Design Spec)` - 10 edges

## Surprising Connections (you probably didn't know these)
- `run_optimize()` --calls--> `select_best_trial()`  [EXTRACTED]
  dsp_inverse.py → relationships.py
- `write_report()` --calls--> `compute_centroids()`  [EXTRACTED]
  dsp_inverse.py → relationships.py
- `write_report()` --calls--> `compute_V()`  [EXTRACTED]
  dsp_inverse.py → relationships.py
- `write_report()` --calls--> `L_guard_value()`  [EXTRACTED]
  dsp_inverse.py → relationships.py
- `cmd_relationships_baseline()` --calls--> `build_baseline_table()`  [EXTRACTED]
  dsp_inverse.py → relationships.py

## Import Cycles
- None detected.

## Communities (18 total, 2 thin omitted)

### Community 0 - "test_relationships.py"
Cohesion: 0.08
Nodes (50): fixture, build_baseline_table(), _centroid(), collapse_gate(), compute_centroids(), compute_distance_matrix(), compute_V(), freeze_table() (+42 more)

### Community 1 - "dsp_inverse.py"
Cohesion: 0.08
Nodes (39): bestsplit_for(), cmd_baseline(), cmd_optimize(), cmd_relationships_baseline(), cmd_reproduce(), Dataset, decode_wav(), detect_transient_window() (+31 more)

### Community 2 - "2. Paper-by-Paper Concepts, Formulas & Assertions"
Cohesion: 0.09
Nodes (22): 1. Executive Summary & Project Mapping, 2.10 Bauer — Galois Actions on SL₂(ℤ) Representations, 2.11 Alvarez-Gaume–Marino — Softly Broken N=2 QCD, 2.12 Luck — Billiards / Aperiodic Tilings, 2.13 Gutzwiller — Arithmetic Chaos, 2.14 Berry — Talbot Effect / Fractals, 2.15 Lesage–Saleur — TBA / Landauer–Buttiker Conductance, 2.16 Kontsevich — Lyapunov / Hodge (+14 more)

### Community 3 - "dsp_chart.py"
Cohesion: 0.18
Nodes (16): ChartHandler, cmd_add(), cmd_clear(), cmd_serve(), copy_and_add(), ensure_dirs(), infer_group(), load_labels() (+8 more)

### Community 4 - "DSP Arena — TV-Game Verdict Screen (Design Spec)"
Cohesion: 0.11
Nodes (17): 0. The three decisions (with evidence), 1. Golden rule (NEVER-RULE, user-corrected 2026-08-09), 2. Data flow (already verified in dsp_chart.py, read 2026-08-11), 3. The TV-game surface (3 screens, one big answer each), 4. Honesty layer (MANDATORY on every verdict — user LAW: never fabricate), 5. Verification anchors (the build is WRONG until these reproduce), 6. Implementation plan (when approved), 7. What an "app" WOULD change (honest) — for later (+9 more)

### Community 5 - "law_cards.py"
Cohesion: 0.23
Nodes (14): cards(), gauss_sum_magnitude(), _legendre(), main(), nice_beta(), law_cards.py - pure mathematical LAW relationship cards, inlined as JSON. Each…, All LAW cards with their pinned reference value and tolerance., Re-derive every computable value; return an honest verdict per card (R5). (+6 more)

### Community 7 - "DSP Inverse Optimizer - Report"
Cohesion: 0.18
Nodes (10): 25-split p-value table (winning params), DSP Inverse Optimizer - Report, Feature importance (one-at-a-time param sensitivity), Fisher ratio per feature, `hop_length`: 4096 -> 2048, Mahalanobis centroid distance (per feature-pair, focus split), Mathematical relationship (parameter -> feature -> separation), `n_fft`: 1024 -> 2048 (+2 more)

### Community 8 - "DSP Inverse Optimizer - Report"
Cohesion: 0.18
Nodes (10): 25-split p-value table (winning params), DSP Inverse Optimizer - Report, Feature importance (one-at-a-time param sensitivity), Fisher ratio per feature, `hop_length`: 2048 -> 1024, Mahalanobis centroid distance (per feature-pair, focus split), Mathematical relationship (parameter -> feature -> separation), `n_fft`: 4096 -> 2048 (+2 more)

### Community 9 - "DSP Inverse Optimizer - Report"
Cohesion: 0.18
Nodes (10): 25-split p-value table (winning params), DSP Inverse Optimizer - Report, Feature importance (one-at-a-time param sensitivity), Fisher ratio per feature, `hop_length`: 512 -> 256, Mahalanobis centroid distance (per feature-pair, focus split), Mathematical relationship (parameter -> feature -> separation), `n_fft`: 1024 -> 2048 (+2 more)

### Community 10 - "DSP Arena — Project Memory"
Cohesion: 0.18
Nodes (10): 0. Resume protocol, 1. Identity & golden rule, 2. Locked decisions (do not re-decide), 3. Verification anchors (build is WRONG until these reproduce), 4. Pitfalls (verified, do not relearn), 5. Data inventory, 6. Canonical commands, 7. Open / future (designs approved, not implemented) (+2 more)

### Community 11 - "DSP Arena — Interactive Acoustic Event Chart"
Cohesion: 0.18
Nodes (10): Browser view, DSP Arena — Interactive Acoustic Event Chart, Features, Frozen JSON schema (v2), Known properties, Quick start, Relationship system, Running the browser app (+2 more)

### Community 12 - "ref_dsp.py"
Cohesion: 0.39
Nodes (7): decode_wav(), max_rms_frame(), measure(), stdlib WAV decode — mirrors JS wavDecode (fmt 1/3/0xFFFE, 8/16/24/32, stereo…, Mirror of JS trimWav: frame power (2048/512), threshold = max*10^(-top_db/10)., resample_linear(), trim_wav()

### Community 13 - "DSP Inverse Optimizer - Report"
Cohesion: 0.29
Nodes (6): 25-split p-value table (winning params), DSP Inverse Optimizer - Report, Feature importance (one-at-a-time param sensitivity), Fisher ratio per feature, Mahalanobis centroid distance (per feature-pair, focus split), Mathematical relationship (parameter -> feature -> separation)

### Community 14 - "DSP Inverse Optimizer - Report"
Cohesion: 0.29
Nodes (6): 25-split p-value table (winning params), DSP Inverse Optimizer - Report, Feature importance (one-at-a-time param sensitivity), Fisher ratio per feature, Mahalanobis centroid distance (per feature-pair, focus split), Mathematical relationship (parameter -> feature -> separation)

## Knowledge Gaps
- **84 isolated node(s):** `0. Resume protocol`, `1. Identity & golden rule`, `2. Locked decisions (do not re-decide)`, `3. Verification anchors (build is WRONG until these reproduce)`, `4. Pitfalls (verified, do not relearn)` (+79 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_baseline_table()` connect `test_relationships.py` to `dsp_inverse.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `compute_centroids()` connect `test_relationships.py` to `dsp_inverse.py`, `._score`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `L_guard_value()` connect `test_relationships.py` to `dsp_inverse.py`, `._score`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **What connects `0. Resume protocol`, `1. Identity & golden rule`, `2. Locked decisions (do not re-decide)` to the rest of the system?**
  _84 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_relationships.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08272859216255443 - nodes in this community are weakly interconnected._
- **Should `dsp_inverse.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07955596669750231 - nodes in this community are weakly interconnected._
- **Should `2. Paper-by-Paper Concepts, Formulas & Assertions` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._