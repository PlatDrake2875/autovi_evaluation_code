# Federated Training Workflow

> Detailed BPMN diagram for federated PatchCore training with memory bank aggregation.

---

## Complete Federated Training Pipeline

```mermaid
flowchart TB
    START([Start Federated Training]) --> CONFIG

    subgraph CONFIG["Configuration"]
        C1["Load Federated Config"] --> C2["Set Random Seeds"]
        C2 --> C3["Initialize Logging"]
        C3 --> C4["Load Data Partitions\n(IID or Category-based)"]
    end

    subgraph INIT["Initialization"]
        C4 --> I1["Create 5 Clients"]
        I1 --> I2["Load Shared Backbone\n(WideResNet-50-2)"]
        I2 --> I3["Initialize Server\nAggregator"]
    end

    subgraph CLIENTTRAIN["Parallel Client Training"]
        I3 --> CT1["Server: Request\nLocal Features"]

        CT1 --> CT2["Client 1:\nExtract Features"]
        CT1 --> CT3["Client 2:\nExtract Features"]
        CT1 --> CT4["Client 3:\nExtract Features"]
        CT1 --> CT5["Client 4:\nExtract Features"]
        CT1 --> CT6["Client 5:\nExtract Features"]

        CT2 --> CS1["Client 1:\nLocal Coreset"]
        CT3 --> CS2["Client 2:\nLocal Coreset"]
        CT4 --> CS3["Client 3:\nLocal Coreset"]
        CT5 --> CS4["Client 4:\nLocal Coreset"]
        CT6 --> CS5["Client 5:\nLocal Coreset"]

        CS1 --> SEND["Send to Server"]
        CS2 --> SEND
        CS3 --> SEND
        CS4 --> SEND
        CS5 --> SEND
    end

    subgraph AGGREGATE["Server Aggregation"]
        SEND --> A1["Receive All\nLocal Coresets"]
        A1 --> A2["Concatenate:\nall_features = concat(coresets)"]
        A2 --> A3["Apply Global\nCoreset Selection"]
        A3 --> A4["Build Global\nMemory Bank"]
    end

    subgraph BROADCAST["Broadcast Global Model"]
        A4 --> B1["Send Global Bank\nto All Clients"]
        B1 --> B2["Clients Update\nLocal Memory"]
    end

    subgraph SAVE["Save Artifacts"]
        B2 --> S1["Save Global\nMemory Bank"]
        S1 --> S2["Save Client\nStatistics"]
        S2 --> S3["Log Summary"]
    end

    S3 --> END([Training Complete])

    style CONFIG fill:#e1f5fe
    style INIT fill:#e8f5e9
    style CLIENTTRAIN fill:#fff3e0
    style AGGREGATE fill:#fce4ec
    style BROADCAST fill:#f3e5f5
    style SAVE fill:#e0f7fa
```

---

## Client Processing Detail

```mermaid
flowchart TB
    subgraph ClientProcess["Single Client Processing"]
        CP1["Receive Request\nfrom Server"] --> CP2["Load Local\nData Partition"]
        CP2 --> CP3["Initialize\nFeature List"]

        CP3 --> CP4["For Each Image\nin Partition"]
        CP4 --> CP5["Apply Preprocessing:\nResize + Normalize"]
        CP5 --> CP6["Forward Pass\nthrough Backbone"]
        CP6 --> CP7["Extract Layer2+3\nFeatures"]
        CP7 --> CP8["Concatenate +\nReshape to Patches"]
        CP8 --> CP9["Append to\nFeature List"]
        CP9 --> CP10{"More\nImages?"}
        CP10 -->|Yes| CP4
        CP10 -->|No| CP11["Stack All\nPatch Features"]

        CP11 --> CP12["Local Coreset\nSelection (10%)"]
        CP12 --> CP13["Return Coreset\nto Server"]
    end

    style ClientProcess fill:#e8f5e9
```

---

## Aggregation Algorithm Detail

```mermaid
flowchart TB
    subgraph Input["Input Coresets"]
        I1["Client 1: n1 patches"]
        I2["Client 2: n2 patches"]
        I3["Client 3: n3 patches"]
        I4["Client 4: n4 patches"]
        I5["Client 5: n5 patches"]
    end

    subgraph Weight["Weighted Sampling"]
        W1["Compute weights:\nw_i = n_i / total"]
        W2["Sample from each:\ns_i = target * w_i * 2"]
    end

    subgraph Combine["Combine Features"]
        CB1["Concatenate\nsampled features"]
        CB2["Total: ~2 * target_size"]
    end

    subgraph GlobalCoreset["Global Coreset Selection"]
        GC1["Initialize: random patch"]
        GC2["Greedy k-center\niteration"]
        GC3["Select target_size\ndiverse patches"]
    end

    subgraph Output["Output"]
        O1["Global Memory Bank\n(target_size patches)"]
    end

    Input --> Weight --> Combine --> GlobalCoreset --> Output

    style Input fill:#e1f5fe
    style Weight fill:#fff3e0
    style Combine fill:#e8f5e9
    style GlobalCoreset fill:#fce4ec
    style Output fill:#f3e5f5
```

---

## IID vs Category-Based Comparison

```mermaid
flowchart TB
    subgraph IID["IID Partitioning"]
        IID1["All 6 categories\nmixed uniformly"]
        IID2["Each client:\n~20% of each category"]
        IID3["Similar distributions\nacross clients"]
        IID1 --> IID2 --> IID3
    end

    subgraph CAT["Category-Based Partitioning"]
        CAT1["Categories assigned\nto specific clients"]
        CAT2["Client 1: engine_wiring\nClient 2: underbody_*\nClient 3: fasteners\nClient 4: pipe_clip\nClient 5: mixed"]
        CAT3["Non-IID:\nheterogeneous distributions"]
        CAT1 --> CAT2 --> CAT3
    end

    subgraph Expected["Expected Outcomes"]
        E1["IID: Higher performance\n(similar to centralized)"]
        E2["Category: Lower performance\n(realistic industrial scenario)"]
    end

    IID --> Expected
    CAT --> Expected

    style IID fill:#e8f5e9
    style CAT fill:#fff3e0
    style Expected fill:#fce4ec
```

---

## Communication Protocol

```mermaid
sequenceDiagram
    participant S as Server
    participant C as Client (any)

    Note over S,C: Phase 1: Initialization
    S->>C: Send backbone weights (if not pre-loaded)
    S->>C: Send coreset parameters

    Note over S,C: Phase 2: Local Training
    S->>C: REQUEST_FEATURES
    C->>C: Extract features from local data
    C->>C: Build local coreset
    C-->>S: SEND_CORESET(local_features)

    Note over S,C: Phase 3: Aggregation
    S->>S: Aggregate all coresets
    S->>S: Build global memory bank

    Note over S,C: Phase 4: Distribution
    S->>C: SEND_GLOBAL_BANK(global_features)
    C->>C: Update local memory

    Note over S,C: Phase 5: Ready for Inference
```

---

## Error Handling

```mermaid
flowchart TB
    E1["Client Processing"] --> E2{"Success?"}
    E2 -->|Yes| E3["Continue"]
    E2 -->|No| E4{"Error Type?"}

    E4 -->|"OOM"| E5["Reduce batch size\nRetry"]
    E4 -->|"Timeout"| E6["Log warning\nExclude client"]
    E4 -->|"Data Error"| E7["Skip corrupted images\nContinue"]

    E5 --> E1
    E6 --> E8["Aggregate with\nremaining clients"]
    E7 --> E3
    E8 --> E3

    style E4 fill:#ffcdd2
    style E5 fill:#fff9c4
    style E6 fill:#fff9c4
    style E7 fill:#fff9c4
```

---

## Performance Metrics

| Metric | IID Expected | Category Expected |
|--------|--------------|-------------------|
| Training Time | ~15 min | ~15 min |
| Communication | ~1 GB | ~1 GB |
| AUC-sPRO@0.05 | ~0.75 | ~0.70 |
| Performance Gap vs Centralized | -3% | -8% |
