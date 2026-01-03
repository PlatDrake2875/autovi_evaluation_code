# 3. Dataset

## 3.1 AutoVI Dataset Overview

The **Automotive Visual Inspection (AutoVI)** dataset [4] is a genuine industrial dataset developed by Renault Group and Université de technologie de Compiègne. Unlike synthetic benchmarks, AutoVI captures real production line conditions including:

- Variations in lighting and brightness
- Moving components during image capture
- Authentic defect patterns from actual production

The dataset contains **6 object categories** from automotive assembly:

| Category | Description | Image Size |
|----------|-------------|------------|
| engine_wiring | Wire harness connections | 400×400 |
| pipe_clip | Pipe securing clips | 400×400 |
| pipe_staple | Pipe fastening staples | 400×400 |
| tank_screw | Fuel tank mounting screws | 1000×750 |
| underbody_pipes | Underbody pipe assemblies | 1000×750 |
| underbody_screw | Underbody mounting screws | 1000×750 |

## 3.2 Dataset Statistics

**Table 1: AutoVI Dataset Statistics**

| Category | Train (good) | Test Total | Test Good | Test Anomaly | Defect Types |
|----------|--------------|------------|-----------|--------------|--------------|
| engine_wiring | 285 | 607 | 285 | 322 | 4 |
| pipe_clip | 195 | 337 | 195 | 142 | 2 |
| pipe_staple | 191 | 305 | 188 | 117 | 1 |
| tank_screw | 318 | 413 | 318 | 95 | 1 |
| underbody_pipes | 161 | 345 | 161 | 184 | 3 |
| underbody_screw | 373 | 392 | 374 | 18 | 1 |
| **Total** | **1,523** | **2,399** | **1,521** | **878** | **10** |

Training data contains only "good" (non-defective) images, following the unsupervised anomaly detection paradigm where models learn normality from defect-free samples.

## 3.3 Defect Types

Defects are categorized into:

- **Structural anomalies**: Physical damage, misalignment, incorrect assembly (e.g., "fastening" in engine_wiring)
- **Logical anomalies**: Missing or misplaced components (e.g., "missing" in tank_screw)

Ground truth annotations are provided as pixel-level segmentation masks, enabling evaluation of both detection and localization performance.

## 3.4 Preprocessing Pipeline

Our preprocessing pipeline follows the evaluation code specifications:

1. **Resizing**: 400×400 for small objects, 1000×750 for large objects
2. **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Tensor conversion**: PyTorch format [C, H, W]

## 3.5 Federated Data Partitioning

We experiment with two partitioning strategies:

**IID Partitioning**: Random uniform distribution across 5 clients (~305 images each, all categories mixed).

**Category-based Partitioning**: Realistic simulation where each client represents a different production line:
- Client 1: engine_wiring (Engine Assembly)
- Client 2: underbody_pipes + underbody_screw (Underbody Line)
- Client 3: tank_screw + pipe_staple (Fastener Station)
- Client 4: pipe_clip (Clip Inspection)
- Client 5: Mixed sample from all categories (Quality Control)

This non-IID distribution reflects real industrial scenarios where different facilities handle different components.
