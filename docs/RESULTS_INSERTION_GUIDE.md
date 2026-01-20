# Results Insertion Guide

Quick reference for updating documentation once experimental results are available.

---

## Timeline

**Current Status**: PatchCore implementation in progress
**Next Step**: Complete model, train 6 independent clients
**Result Population**: After baseline experiments complete

---

## What Results Are Needed?

### Primary Results: Per-Client Baseline Performance

For each of 6 clients/categories:
- **AUC-sPRO** at 4 FPR thresholds: 0.01, 0.05, 0.1, 0.3
- **AUC-ROC**: Image-level classification score
- **Training time**: (optional) How long each client takes
- **Memory usage**: (optional) Storage for memory bank

**6 Clients to Evaluate**:
1. engine_wiring (285 train, 607 test)
2. pipe_clip (195 train, 337 test)
3. pipe_staple (191 train, 305 test)
4. tank_screw (318 train, 413 test)
5. underbody_pipes (161 train, 345 test)
6. underbody_screw (373 train, 392 test)

---

## Files to Update (In Order)

### 1. Markdown Technical Report Sections

**File**: `docs/technical-report/sections/05-baseline-results.md`

**Table to Update**: Table 3 (Section 5.2)

```markdown
| Object | @FPR=0.01 | @FPR=0.05 | @FPR=0.1 | @FPR=0.3 | AUC-ROC |
|--------|-----------|-----------|----------|----------|---------|
| engine_wiring | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| pipe_clip | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| pipe_staple | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| tank_screw | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| underbody_pipes | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| underbody_screw | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| **Mean** | **[VALUE]** | **[VALUE]** | **[VALUE]** | **[VALUE]** | **[VALUE]** |
```

**Status Line to Replace**:
- Find: `**Status**: Awaiting PatchCore baseline experiments on centralized aggregated dataset (all 6 categories combined).`
- Replace with: `**Status**: Results collected [DATE]. Mean AUC-sPRO@0.05: [VALUE]. See detailed results in Table 3.`

---

### 2. Preliminary Results Section

**File**: `docs/technical-report/sections/07-preliminary-results.md`

**Section 7.1**: Update implementation status table

```markdown
| Component | Status | Notes |
|-----------|--------|-------|
| Data Loader | ✓ Complete | All 6 categories loaded and partitioned |
| PatchCore Model | ✓ Complete | Feature extraction and memory bank ready |
| Client Training | ✓ Complete | All 6 clients trained independently |
| Baseline Evaluation | ✓ Complete | Results collected [DATE] |
| Stage 1 Results | ✓ Complete | See Table 4 below |
```

**Table 4**: Replace with actual results:

```markdown
| Client | Category | Train Images | @FPR=0.01 | @FPR=0.05 | @FPR=0.1 | AUC-ROC |
|--------|----------|--------------|-----------|-----------|----------|---------|
| 1 | engine_wiring | 285 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| 2 | pipe_clip | 195 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| 3 | pipe_staple | 191 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| 4 | tank_screw | 318 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| 5 | underbody_pipes | 161 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| 6 | underbody_screw | 373 | [VALUE] | [VALUE] | [VALUE] | [VALUE] |
| **Mean** | **-** | **1,523** | **[VALUE]** | **[VALUE]** | **[VALUE]** | **[VALUE]** |
```

**Section 7.3-7.6**: Replace placeholder comments with actual analysis

---

### 3. LaTeX Technical Report

**File**: `docs/technical-report/Stage1_Technical_Report.tex`

**Section 5.2**: Update Table 4

Find:
```latex
\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Client} & \textbf{Category} & \textbf{Train} & \textbf{@FPR=0.01} & \textbf{@FPR=0.05} & \textbf{@FPR=0.1} & \textbf{AUC-ROC} \\
\midrule
1 & engine\_wiring & 285 & \% TODO & \% TODO & \% TODO & \% TODO \\
...
```

Replace with actual values (keep same format):
```latex
1 & engine\_wiring & 285 & 0.XX & 0.XX & 0.XX & 0.XX \\
```

**Update status text** (Section 5.1):
- Find: `**Status**: Awaiting PatchCore baseline experiments...`
- Replace with implementation completion note

---

### 4. Beamer Presentation

**File**: `docs/presentation/Stage1_Presentation.tex`

**Slide 7**: "Stage 1 Results - Placeholder"

Update table with actual results:
```latex
\begin{table}[H]
\tiny
\begin{tabular}{lcccccc}
...
C1: engine\_wiring & 285 & [VALUE] & [VALUE] & [VALUE] & [VALUE] & [VALUE] \\
...
```

**Slide 8**: Update implementation status colors
- Change `\textcolor{red}{\textbf{Pending}}` to `\textcolor{darkgreen}{\checkmark Complete}`

---

## Visualization Placeholders to Fill

### Figure 1: Anomaly Maps (6 images, one per category)

**Where**:
- Markdown: `sections/05-baseline-results.md` (Section 5.4)
- LaTeX Report: `Stage1_Technical_Report.tex` (Section 5.3)

**What to create**:
- 2×3 grid (or similar layout) with sample images
- One category per image
- Show: input image + anomaly heatmap overlay
- Include one good sample and one anomalous sample per category
- Add captions explaining detection quality

**Suggested format**:
- PNG or PDF for LaTeX inclusion
- ~600×600 pixel per image, 72 DPI
- Save as: `docs/assets/Fig1_AnomalyMaps.png`

---

### Figure 2: System Architecture Diagram

**Where**:
- LaTeX Report: `Stage1_Technical_Report.tex` (Section 6.2)
- Presentation: `Stage1_Presentation.tex` (Slide 5)

**What to show**:
- Central server component (top)
- 6 client nodes below (color-coded by category)
- Data flow arrows (input: images, output: memory banks)
- Optional: Stage 2 aggregation paths (dashed lines for future)

**Can be created with**:
- TikZ (in LaTeX directly)
- Draw.io / Lucidchart (export as PNG/PDF)
- PowerPoint (export as PNG)
- Any diagramming tool

**Suggested format**:
- PDF for vector scalability (best)
- PNG as fallback
- Save as: `docs/assets/Fig2_Architecture.pdf` or `.png`

---

### Figure 3: FPR-sPRO Curves

**Where**:
- Markdown: `sections/07-preliminary-results.md` (Section 7.3)
- LaTeX Report: `Stage1_Technical_Report.tex` (implied in methodology)

**What to plot**:
- X-axis: False Positive Rate (0 to 1)
- Y-axis: sPRO score (0 to 1)
- 6 curves (one per category) or grouped by type
- Include legend with category names
- Optional: Add mean curve across all categories

**Suggested tool**:
- Python matplotlib with results CSV
- Export as PNG or PDF
- Save as: `docs/assets/Fig3_FPRsPRO_Curves.png`

---

### Figure 4: Category Performance Comparison

**Where**:
- Presentation: `Stage1_Presentation.tex` (Slide 7, implied)

**What to plot**:
- Bar chart comparing AUC-sPRO@0.05 across 6 categories
- Optional: Add error bars for variance
- Sort by performance (descending)
- Color-code bars by category type (small vs large objects)

**Suggested tool**:
- Python matplotlib or seaborn
- Export as PNG
- Save as: `docs/assets/Fig4_CategoryComparison.png`

---

## Testing Before Population

1. **Verify numbers**:
   - Check that AUC scores are in [0, 1] range
   - Verify sums and means are correct
   - Confirm all 4 FPR thresholds present

2. **Check formatting**:
   - Markdown tables align properly
   - LaTeX compiles without errors: `pdflatex Stage1_Technical_Report.tex`
   - Beamer compiles without errors: `pdflatex Stage1_Presentation.tex`

3. **Validate figures**:
   - All image files exist and are accessible
   - LaTeX paths are correct relative to `.tex` file
   - File sizes reasonable (< 2MB each)

---

## Updating From CSV Results

If results are in CSV format, here's a Python snippet to generate tables:

```python
import pandas as pd

# Read results
df = pd.read_csv('baseline_results.csv')
# Expected columns: category, fpr_001, fpr_005, fpr_01, fpr_03, auc_roc

# Generate Markdown table
print(df.to_markdown(index=False))

# Or for LaTeX
print(df.to_latex(index=False))
```

---

## Checklist for Complete Results Population

- [ ] All 6 clients have baseline results (AUC-sPRO @ 4 FPR + AUC-ROC)
- [ ] Table 3 in markdown (sections/05-baseline-results.md) updated
- [ ] Table 4 in markdown (sections/07-preliminary-results.md) updated
- [ ] Status indicators changed in implementation status tables
- [ ] LaTeX report Table 4 (Stage1_Technical_Report.tex) updated
- [ ] Beamer Slide 7 results table updated
- [ ] Beamer Slide 8 implementation status colors updated
- [ ] Figure 1 (anomaly maps) created and referenced
- [ ] Figure 2 (architecture) created and referenced (if new)
- [ ] Anomaly map sections have actual figure links
- [ ] All "% TODO" comments replaced or status updated
- [ ] LaTeX report compiles without errors
- [ ] Beamer presentation compiles without errors
- [ ] Markdown files render correctly with tables
- [ ] Cross-references between documents are consistent

---

## After Population

1. **Generate PDFs**:
   ```bash
   cd docs/technical-report
   pdflatex Stage1_Technical_Report.tex
   pdflatex Stage1_Technical_Report.tex  # Run twice for references

   cd ../presentation
   pdflatex Stage1_Presentation.tex
   pdflatex Stage1_Presentation.tex
   ```

2. **Create combined markdown report** (optional):
   ```bash
   cd technical-report/sections
   cat 01-introduction.md 02-related-work.md 03-dataset.md \
       04-methodology.md 05-baseline-results.md 06-federated-setup.md \
       07-preliminary-results.md 08-next-steps.md > full_report.md
   ```

3. **Verify final documents**:
   - Open PDFs and check formatting
   - Verify all tables and figures render
   - Test any hyperlinks
   - Check page count matches expectations (LaTeX report: 18-20 pages)

---

## Notes

- **LaTeX compilation**: First run creates cross-references, second run resolves them
- **PDF size**: If LaTeX output is > 10MB, may need to optimize images
- **Beamer notes**: Add speaker notes under slides as needed (optional)
- **Version control**: Commit updated documents with message like "Stage 1: Populate baseline results"

---

*This guide ensures consistent documentation updates across all three documentation formats (Markdown sections, LaTeX report, Beamer presentation). Follow the order listed for best results.*
