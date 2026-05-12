name: introduction-protection
description: Introduction §1 (lines 62–77) was expert-written and is PROTECTED. All editing in other sections must align with its narrative frame. Never rewrite, add citations, or reorder Intro paragraphs.

paths:
  - "main.tex"

## Narrative Threads from Intro — Other Sections Must Echo These

### Thread 1: Problem Framing
> "Understanding how complex systems evolve... requires uncovering not only statistical associations but also the underlying causal mechanisms"
- All sections emphasize CAUSALITY over correlation. Use "causal link/mechanism", never "association/pattern".

### Thread 2: Three Gaps — Every Results Subsection Must Close One
1. **Scale**: Most methods focus on small-to-medium systems (N < hundreds)
   → Closed by climate (N=10,512), traffic scaling up to GLA (N=3,834)
2. **Static assumption**: Causal graphs assumed fixed over time
   → Closed by Beijing shockwave analysis, dynamic causal intensity snapshots
3. **Domain specificity**: Handcrafted model classes for specific domains
   → Closed by cross-modal EVCDP (zero-shot dual-modality), synthetic-to-real transfer

### Thread 3: Core Thesis
> "Causal inference in complex systems must be fundamentally reframed as a geometry-aware and time-varying learning problem"
- Every methodological choice connects to one of two pillars: **geometry-aware** or **time-varying**.

### Thread 4: Contribution Framing
> "from correlation-based prediction toward mechanistic understanding"
- Discussion closing paragraph must echo this language. Maintain broader vision, not narrow technical claims.

## What "Protected" Means in Practice
- ❌ No rewriting for "flow" or "clarity"
- ❌ No adding new citations or examples  
- ❌ No changing paragraph order
- ✅ Results/Methodology/Discussion use identical terminology to Intro
- ✅ No section contradicts an Intro claim
