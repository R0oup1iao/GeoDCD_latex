# Quick Reference — Key Numbers & Facts

## Scale Numbers (Memorize)
- Climate SLP: **N = 10,512** nodes (ERA5 at 2.5°×2.5°)
- Traffic max: GLA **N = 3,834** sensors (Caltrans PeMS)
- Shenzhen EVCDP: **N = 494** (247 traffic + 247 charging)
- Synthetic Finance: **N = 40** (synthetic financial time-series)

## Performance Highlights
| Dataset | Metric | GeoDCD | Runner-up | Reduction |
|---------|--------|--------|-----------|-----------|
| Lorenz-96 | F1 | **0.99** | Causalformer 0.95 | SHD −36.8% |
| Cluster-Lorenz | F1 | **0.95** | UnCLE 0.89 | — |
| Finance (adversarial geo) | F1 / AUROC | 0.78 / **0.96** | Causalformer 0.72 | — |

## Method Components → Why They Exist
- Hierarchical Geometric Pooling → O(N²) → near-linear, prunes impossible long-range links
- Jacobian sensitivity S_{i←j}(t) = ‖∂x̂/∂x_t‖_F → extracts instantaneous causal strength (the "time-varying" part)
- Basis Decomposition (N_basis=4) → parameter efficiency, shared interaction patterns across channels
- Causal Transformer Pre-LN → temporal modeling with strict causality mask + training stability

## File Map
```
.mnt/e/GeoDCD_latex/
├── main.tex              711 lines (~84KB)
├── ref.bib               Bibliography
├── figures/              ~30 PDF/PNG files
└── .claude/
    ├── CLAUDE.md         Core instructions (≤200 lines)
    └── rules/
        ├── introduction-protection.md  §1 is sacred
        └── results-alignment.md        Results must close Intro gaps
```
