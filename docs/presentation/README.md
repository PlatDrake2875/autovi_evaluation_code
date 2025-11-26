# Presentation Materials - Stage 1

> **Federated Learning for Industrial Anomaly Detection**
> Short Presentation for Stage 1 Deliverables

---

## Presentation Overview

| Aspect | Details |
|--------|---------|
| Duration | 10-15 minutes |
| Format | Slide deck (PowerPoint/Google Slides/PDF) |
| Audience | Course instructors, peers |
| Focus | Dataset, model, preliminary results, next steps |

---

## Slide Deck Structure

See [slides-outline.md](slides-outline.md) for detailed slide content.

| Slide | Title | Duration | Speaker |
|-------|-------|----------|---------|
| 1 | Title & Team | 30s | All |
| 2 | Problem: Privacy in Industrial Inspection | 1.5min | Member 1 |
| 3 | AutoVI Dataset Overview | 1.5min | Member 1 |
| 4 | PatchCore Architecture | 2min | Member 2 |
| 5 | Federated Setup (5 Clients) | 2min | Member 2 |
| 6 | Results: Baseline Performance | 1.5min | Member 3 |
| 7 | Results: Federated Comparison | 1.5min | Member 3 |
| 8 | Analysis & Key Findings | 1.5min | Member 3 |
| 9 | Stage 2 Roadmap | 1min | All |
| 10 | Q&A | 2-3min | All |

---

## Presentation Assets

### Required Figures

| Figure | Description | Source |
|--------|-------------|--------|
| Fig 1 | AutoVI sample images (6 categories) | Dataset |
| Fig 2 | PatchCore architecture diagram | docs/phases/phase-2 |
| Fig 3 | Federated system architecture | docs/phases/phase-3 |
| Fig 4 | Results comparison chart | Experiments |
| Fig 5 | FPR-sPRO curves | Experiments |

### Required Tables

| Table | Description |
|-------|-------------|
| Tab 1 | Dataset statistics |
| Tab 2 | Centralized vs Federated results |

---

## Speaker Notes

See [speaker-notes.md](speaker-notes.md) for detailed talking points.

---

## Preparation Checklist

- [ ] Create slide deck (PowerPoint/Google Slides)
- [ ] Generate all figures from experiments
- [ ] Populate results tables with actual data
- [ ] Write speaker notes
- [ ] Practice timing (aim for 12 minutes)
- [ ] Prepare for Q&A (anticipate questions)
- [ ] Export to PDF as backup

---

## Tips for Presentation

1. **Start with the problem**: Why does industrial anomaly detection need FL?
2. **Show, don't tell**: Use diagrams and visualizations
3. **Be honest about limitations**: Acknowledge the performance gap
4. **Connect to Stage 2**: End with clear roadmap

---

## Q&A Preparation

### Anticipated Questions

1. **Why PatchCore over other methods?**
   - State-of-the-art performance
   - Natural fit for memory bank aggregation
   - Interpretable anomaly maps

2. **Why only 5 clients?**
   - Matches project guidelines (3-5 clients)
   - Sufficient for demonstrating IID vs non-IID effects
   - Scalable to more clients

3. **How will DP affect performance?**
   - Expected 5-10% additional degradation
   - Trade-off analysis planned for Stage 2

4. **Is one communication round enough?**
   - PatchCore is non-iterative (feature bank, not gradients)
   - Could explore iterative refinement in Stage 2

5. **How do you handle client dropout?**
   - Current implementation assumes all clients participate
   - Future work could add fault tolerance
