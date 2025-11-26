# Development Phases

This project is organized into **4 sequential phases** for Stage 1 development.

---

## Phase Overview

```mermaid
flowchart LR
    P1["Phase 1\nData Preparation"] --> P2["Phase 2\nBaseline Model"]
    P2 --> P3["Phase 3\nFederated Setup"]
    P3 --> P4["Phase 4\nEvaluation"]
    P4 --> D["Deliverables\n(Report + Code + Presentation)"]

    style P1 fill:#e1f5fe
    style P2 fill:#e8f5e9
    style P3 fill:#fff3e0
    style P4 fill:#fce4ec
    style D fill:#f3e5f5
```

---

## Phase Summary

| Phase | Focus | Key Outputs |
|-------|-------|-------------|
| [Phase 1](phase-1-data-preparation/README.md) | Data Preparation | Dataset loader, FL partitions, EDA |
| [Phase 2](phase-2-baseline-model/README.md) | Baseline Model | PatchCore centralized training |
| [Phase 3](phase-3-federated-setup/README.md) | Federated Setup | FL client/server, memory aggregation |
| [Phase 4](phase-4-evaluation/README.md) | Evaluation | Metrics, comparison, analysis |

---

## Timeline and Dependencies

```mermaid
gantt
    title Stage 1 Development Phases
    dateFormat  YYYY-MM-DD
    section Phase 1
    Data Acquisition           :p1a, 2024-01-01, 2d
    Preprocessing Pipeline     :p1b, after p1a, 2d
    FL Partitioning           :p1c, after p1b, 2d

    section Phase 2
    PatchCore Implementation   :p2a, after p1c, 3d
    Centralized Training       :p2b, after p2a, 2d

    section Phase 3
    FL Client Implementation   :p3a, after p2b, 3d
    FL Server & Aggregation    :p3b, after p3a, 2d
    FL Experiments            :p3c, after p3b, 2d

    section Phase 4
    Evaluation Pipeline        :p4a, after p3c, 2d
    Comparison Analysis        :p4b, after p4a, 2d
    Documentation             :p4c, after p4b, 2d
```

---

## Team Member Assignments

| Phase | Primary Lead | Support |
|-------|--------------|---------|
| Phase 1 | Data & Experiment Lead | All |
| Phase 2 | Modeling Lead | Data Lead |
| Phase 3 | Modeling Lead | Evaluation Lead |
| Phase 4 | Evaluation Lead | All |

---

## Cross-Evaluation Protocol

Per project guidelines, the team must implement **cross-evaluation**:

1. **Each member** trains a model variant on their assigned data split
2. **All models** are tested by other members on their local test partitions
3. **Results** are aggregated with inter-member comparison tables

```mermaid
flowchart TB
    subgraph Training["Model Training"]
        M1["Member 1\nTrain on Split A"]
        M2["Member 2\nTrain on Split B"]
        M3["Member 3\nTrain on Split C"]
    end

    subgraph CrossEval["Cross-Evaluation"]
        M1 -->|"Model A"| E1["Test on Split B"]
        M1 -->|"Model A"| E2["Test on Split C"]
        M2 -->|"Model B"| E3["Test on Split A"]
        M2 -->|"Model B"| E4["Test on Split C"]
        M3 -->|"Model C"| E5["Test on Split A"]
        M3 -->|"Model C"| E6["Test on Split B"]
    end

    subgraph Results["Results Aggregation"]
        E1 & E2 & E3 & E4 & E5 & E6 --> R["Comparison Tables"]
    end

    style Training fill:#e3f2fd
    style CrossEval fill:#fff8e1
    style Results fill:#e8f5e9
```
