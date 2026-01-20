# Evaluation Workflow

> Detailed BPMN diagram for model evaluation and comparison analysis.

---

## Complete Evaluation Pipeline

```mermaid
flowchart TB
    START([Start Evaluation]) --> LOAD

    subgraph LOAD["Load Models"]
        L1["Load Centralized\nPatchCore Model"]
        L2["Load Federated (IID)\nMemory Bank"]
        L3["Load Federated (Category)\nMemory Bank"]
    end

    subgraph ANOMALY["Anomaly Map Generation"]
        LOAD --> A1["For Each Model"]
        A1 --> A2["For Each Object\nCategory"]
        A2 --> A3["For Each Test\nImage"]
        A3 --> A4["Extract Features\n(WideResNet-50)"]
        A4 --> A5["Query Memory Bank\nNearest Neighbor"]
        A5 --> A6["Compute Distance\nMap"]
        A6 --> A7["Upsample to\nOriginal Size"]
        A7 --> A8["Normalize to\n[0, 255]"]
        A8 --> A9["Save as PNG"]
        A9 --> A10{"More\nImages?"}
        A10 -->|Yes| A3
        A10 -->|No| A11{"More\nObjects?"}
        A11 -->|Yes| A2
        A11 -->|No| A12{"More\nModels?"}
        A12 -->|Yes| A1
        A12 -->|No| METRICS
    end

    subgraph METRICS["Metrics Computation"]
        M1["For Each Model"]
        M1 --> M2["For Each Object"]
        M2 --> M3["Load Ground Truth\nMasks"]
        M3 --> M4["Load Generated\nAnomaly Maps"]
        M4 --> M5["Initialize\nMetricsAggregator"]
        M5 --> M6["Binary Threshold\nRefinement"]
        M6 --> M7["Compute sPRO\nper Defect"]
        M7 --> M8["Compute FPR\nat Each Threshold"]
        M8 --> M9["Build FPR-sPRO\nCurve"]
        M9 --> M10["Compute AUC-sPRO\n@ [0.01, 0.05, 0.1, 0.3, 1.0]"]
        M10 --> M11["Compute Image-Level\nAUC-ROC"]
        M11 --> M12["Save Metrics\nJSON"]
        M12 --> M13{"More\nObjects?"}
        M13 -->|Yes| M2
        M13 -->|No| M14{"More\nModels?"}
        M14 -->|Yes| M1
        M14 -->|No| COMPARE
    end

    subgraph COMPARE["Comparison Analysis"]
        C1["Load All Metrics\nJSON Files"]
        C1 --> C2["Build Comparison\nDataFrame"]
        C2 --> C3["Compute Performance\nGaps"]
        C3 --> C4["Generate Tables:\n- Per-Object\n- Per-Defect Type\n- Aggregate"]
        C4 --> C5["Generate Plots:\n- FPR-sPRO Curves\n- Bar Charts\n- Heatmaps"]
        C5 --> C6["Statistical Tests:\n- Paired t-test\n- Wilcoxon\n- Effect Size"]
        C6 --> C7["Generate Summary\nReport"]
    end

    C7 --> END([Evaluation Complete])

    style LOAD fill:#e1f5fe
    style ANOMALY fill:#e8f5e9
    style METRICS fill:#fff3e0
    style COMPARE fill:#fce4ec
```

---

## Anomaly Map Generation Detail

```mermaid
flowchart LR
    subgraph Input["Input"]
        I1["Test Image\n[3, H, W]"]
    end

    subgraph Extract["Feature Extraction"]
        E1["WideResNet-50\n(frozen)"]
        E2["Layer 2+3\nFeatures"]
        E3["Patch Grid\n[N, 1536]"]
    end

    subgraph Query["Memory Query"]
        Q1["FAISS Index\n(Memory Bank)"]
        Q2["k-NN Search\n(k=1)"]
        Q3["Distance Values"]
    end

    subgraph Output["Output"]
        O1["Reshape to\n[H/8, W/8]"]
        O2["Bilinear Upsample\n[H, W]"]
        O3["Normalize\n[0, 255]"]
        O4["Save PNG"]
    end

    Input --> Extract --> Query --> Output

    style Input fill:#e1f5fe
    style Extract fill:#e8f5e9
    style Query fill:#fff3e0
    style Output fill:#fce4ec
```

---

## Metrics Computation Detail

```mermaid
flowchart TB
    subgraph LoadData["Load Data"]
        LD1["Ground Truth Maps\n(pixel masks)"]
        LD2["Anomaly Maps\n(model output)"]
        LD3["Defects Config\n(JSON)"]
    end

    subgraph ThresholdSearch["Threshold Search"]
        TS1["Initialize threshold\nrange [0, 1]"]
        TS2["Binary search for\noptimal thresholds"]
        TS3["Refine to achieve\nmax_distance = 0.001"]
    end

    subgraph PerThreshold["Per-Threshold Metrics"]
        PT1["For each threshold t"]
        PT2["Binary anomaly map:\npixel > t"]
        PT3["Compute TP, FP, TN, FN\nper defect region"]
        PT4["sPRO = min(TP/Sat, 1)"]
        PT5["FPR = FP / (FP + TN)"]
    end

    subgraph Aggregate["Aggregation"]
        AG1["Mean sPRO across\ndefects"]
        AG2["Build FPR-sPRO\ncurve"]
        AG3["Interpolate at\nFPR limits"]
        AG4["Compute AUC\n(trapezoidal)"]
    end

    subgraph Classification["Image Classification"]
        CL1["max(anomaly_map)\nper image"]
        CL2["Labels: good=0,\nanomaly=1"]
        CL3["ROC curve"]
        CL4["AUC-ROC"]
    end

    LoadData --> ThresholdSearch --> PerThreshold --> Aggregate
    LoadData --> Classification

    style LoadData fill:#e1f5fe
    style ThresholdSearch fill:#e8f5e9
    style PerThreshold fill:#fff3e0
    style Aggregate fill:#fce4ec
    style Classification fill:#f3e5f5
```

---

## Comparison Report Generation

```mermaid
flowchart TB
    subgraph Input["Input Metrics"]
        I1["centralized/\n*.json"]
        I2["federated_iid/\n*.json"]
        I3["federated_category/\n*.json"]
    end

    subgraph Tables["Generate Tables"]
        T1["Per-Object Comparison\n(6 rows x 3 methods)"]
        T2["Per-Defect Type\n(structural vs logical)"]
        T3["Aggregate Summary\n(mean, std)"]
        T4["Performance Gap\n(% difference)"]
    end

    subgraph Plots["Generate Plots"]
        P1["FPR-sPRO Curves\n(per object)"]
        P2["Bar Chart\n(method comparison)"]
        P3["Heatmap\n(object x method)"]
        P4["Box Plot\n(distribution)"]
    end

    subgraph Stats["Statistical Analysis"]
        S1["Paired t-test\n(centralized vs fed)"]
        S2["Wilcoxon signed-rank"]
        S3["Effect size\n(Cohen's d)"]
        S4["Confidence intervals"]
    end

    subgraph Report["Final Report"]
        R1["Markdown Summary"]
        R2["LaTeX Tables"]
        R3["PDF Figures"]
    end

    Input --> Tables --> Report
    Input --> Plots --> Report
    Input --> Stats --> Report

    style Input fill:#e1f5fe
    style Tables fill:#e8f5e9
    style Plots fill:#fff3e0
    style Stats fill:#fce4ec
    style Report fill:#f3e5f5
```

---

## Cross-Evaluation Protocol (Team)

```mermaid
flowchart TB
    subgraph Member1["Member 1 Training"]
        M1T["Train on\nSplit A"]
        M1M["Model A"]
    end

    subgraph Member2["Member 2 Training"]
        M2T["Train on\nSplit B"]
        M2M["Model B"]
    end

    subgraph Member3["Member 3 Training"]
        M3T["Train on\nSplit C"]
        M3M["Model C"]
    end

    subgraph CrossEval["Cross-Evaluation Matrix"]
        CE["Each model tested on\nall 3 splits"]
    end

    subgraph Results["Results Matrix"]
        R1["3x3 Performance\nMatrix"]
        R2["Diagonal: Same-split\nOff-diagonal: Cross-split"]
        R3["Analyze generalization"]
    end

    Member1 --> CrossEval
    Member2 --> CrossEval
    Member3 --> CrossEval
    CrossEval --> Results

    style Member1 fill:#c8e6c9
    style Member2 fill:#fff9c4
    style Member3 fill:#ffccbc
    style CrossEval fill:#e1f5fe
    style Results fill:#fce4ec
```

---

## Output Artifacts Summary

| Artifact | Path | Format |
|----------|------|--------|
| Anomaly maps | `outputs/anomaly_maps/{method}/{object}/` | PNG |
| Per-object metrics | `outputs/metrics/{method}/{object}/metrics.json` | JSON |
| Comparison table | `outputs/reports/comparison_table.csv` | CSV |
| FPR-sPRO curves | `outputs/reports/fpr_spro_curves.pdf` | PDF |
| Statistical analysis | `outputs/reports/statistical_analysis.json` | JSON |
| Summary report | `outputs/reports/summary_report.md` | Markdown |
