# Technical Report - Stage 1

> **Federated Learning for Industrial Anomaly Detection**
> Stage 1: Baseline Development and Minimal Federated Setup

---

## Report Overview

This technical report (3-4 pages) documents Stage 1 deliverables for the "AI for Trustworthy Decision Making" project.

### Report Structure

| Section | Pages | File |
|---------|-------|------|
| 1. Introduction | 0.4 | [01-introduction.md](sections/01-introduction.md) |
| 2. Related Work | 0.5 | [02-related-work.md](sections/02-related-work.md) |
| 3. Dataset | 0.5 | [03-dataset.md](sections/03-dataset.md) |
| 4. Methodology | 0.6 | [04-methodology.md](sections/04-methodology.md) |
| 5. Baseline Results | 0.4 | [05-baseline-results.md](sections/05-baseline-results.md) |
| 6. Federated Setup | 0.4 | [06-federated-setup.md](sections/06-federated-setup.md) |
| 7. Preliminary Results | 0.5 | [07-preliminary-results.md](sections/07-preliminary-results.md) |
| 8. Next Steps | 0.25 | [08-next-steps.md](sections/08-next-steps.md) |
| References | 0.25 | Inline |
| **Total** | **~3.5** | |

---

## Compilation Instructions

### Option 1: Jupyter Notebook Export

The report can be generated from the project notebook:

```bash
jupyter nbconvert --to pdf notebooks/report.ipynb
```

### Option 2: Markdown to PDF

Using pandoc:

```bash
cd docs/technical-report/sections
cat 01-introduction.md 02-related-work.md 03-dataset.md \
    04-methodology.md 05-baseline-results.md 06-federated-setup.md \
    07-preliminary-results.md 08-next-steps.md > full_report.md

pandoc full_report.md -o ../Stage1_Technical_Report.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt
```

### Option 3: LaTeX

Convert markdown sections to LaTeX and compile:

```bash
pandoc sections/*.md -o report.tex
pdflatex report.tex
```

---

## Figures and Tables

### Required Figures

1. **Figure 1**: AutoVI dataset samples (one per object category)
2. **Figure 2**: PatchCore architecture diagram
3. **Figure 3**: Federated architecture with 5 clients
4. **Figure 4**: FPR-sPRO curves comparison

### Required Tables

1. **Table 1**: AutoVI dataset statistics
2. **Table 2**: Model hyperparameters
3. **Table 3**: Centralized baseline results
4. **Table 4**: Federated comparison (IID vs Category-based)

---

## Team Member Contributions

Per project guidelines, clearly attribute contributions:

| Section | Primary Author | Reviewer |
|---------|----------------|----------|
| Introduction | All | All |
| Related Work | Member 2 | Member 1 |
| Dataset | Member 1 | Member 3 |
| Methodology | Member 2 | Member 1 |
| Baseline Results | Member 2 | Member 3 |
| Federated Setup | Member 1 | Member 2 |
| Preliminary Results | Member 3 | All |
| Next Steps | All | All |

---

## Checklist

- [ ] All sections written and reviewed
- [ ] Figures generated and inserted
- [ ] Tables populated with actual results
- [ ] References formatted correctly (3-5 minimum)
- [ ] Page count verified (3-4 pages)
- [ ] Team member contributions documented
- [ ] PDF exported and tested
