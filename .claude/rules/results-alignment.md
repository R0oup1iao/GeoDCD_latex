name: results-alignment
description: Each Results subsection must explicitly close one of the 3 gaps from Introduction (scale / static-graph assumption / domain-specific models). Terminology must match Intro's "geometry-aware + time-varying" framing.

paths:
  - "main.tex"

## Subsection-to-Gap Mapping

| Subsection | Gap Closed | Key Phrase to Use |
|------------|-----------|-------------------|
| §2.1 Framework intro | All three (overview) | "bridges the gap", "geometry-constrained optimization" |
| §2.2 Synthetic benchmarks | Scale + Domain-specific | "dual-mode learner", robustness under adversarial geometry |
| §2.3 Climate (N=10,512) | Scale (crown jewel) | "largest data-driven causal teleconnection analysis to date" |
| §2.4 Traffic scalability | Scale | "near-linear scaling R²=0.992", computational feasibility |
| §2.5 Congestion dynamics | Static-graph assumption | "disentangles regimes that static methods conflate" |
| §2.6 Cross-modal EVCDP | Domain-specific (zero-shot) | "without architectural modification" |
| §2.7 Ablation | Validation of design choices | Not a gap-close, but validates all components |

## Writing Requirements for Each Results Subsection
1. **First paragraph**: State which gap this subsection addresses and how
2. **Results presentation**: Use bold for GeoDCD numbers, not for everything
3. **Connection back to Intro**: End with 1–2 sentences explicitly linking result to the "geometry-aware + time-varying" thesis
4. **No new claims**: Every finding must be foreshadowed in Introduction

## Terminology Lock
- GeoDCD framework components: "Hierarchical Geometric Pooling", "Jacobian-based Dynamic Inference", "Basis Decomposition" (exact names)
- Output types: "static causal graph A ∈ R^{N×N}" and "dynamic causal strength S(t)"
- Never use: "neural network learns correlations", "statistical associations", "pattern recognition"
