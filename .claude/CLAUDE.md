# GeoDCD — Nature Sub-journal Manuscript

## Commands
pdflatex main.tex                    # 编译 PDF
bibtex main                          # 更新参考文献
pdflatex main.tex && pdflatex main.tex  # 完整编译（引用+交叉引用）

## Architecture
- LaTeX article class, NEJM citation style (biblatex)
- `main.tex` — 711 lines, all content in one file
- `ref.bib` — bibliography
- `figures/` — PDF + PNG figures (~30 files)
- `.claude/rules/*.md` — modular writing constraints

## Critical Rules
- **§1 Introduction (lines 62–77): NEVER MODIFY.** Expert-written. All other sections must align with its narrative frame.
- Every Results subsection must explicitly close one of the 3 gaps from Intro: (i) scale, (ii) static-graph assumption, (iii) domain-specific models.
- Core thesis: causal inference = geometry-aware + time-varying learning problem. Echo this in every section.
- Terminology consistency: "causal discovery" not "correlation analysis"; "time-varying/instantaneous" for S(t), never "static graph".

## Watch out
- **Node counts**: Verify consistency across all sections. Complete list:
  - Synthetic: VAR (N=128), Lorenz-96 (configurable), Cluster-Lorenz, Finance synthetic time-series (N=40)
  - Climate: ERA5 SLP grid ($2.5^\circ\times2.5^\circ$) → N=10,512 nodes
  - Traffic: San Diego PeMS D8 (N=716), Beijing ring/arterial (N=513), Greater Bay Area (N=2,352), Greater Los Angeles (N=3,834)
  - Cross-modal EVCDP (Shenzhen): Dual-modality network of N=494 nodes total (derived from 247 TAZs $\times$ 2 modalities: traffic + energy).
- When rewriting any section, check that no new claims appear that weren't foreshadowed in Introduction.
- Discussion limitations (5 items) must correspond to actual method constraints — not invented weaknesses.
