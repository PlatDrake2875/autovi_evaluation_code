# 1. Introduction

## 1.1 Motivation

Industrial visual inspection is critical for ensuring product quality in manufacturing environments. Automotive production lines, in particular, require robust defect detection systems capable of identifying anomalies that may compromise vehicle safety and reliability. Traditional centralized machine learning approaches face significant challenges in real-world industrial deployments:

1. **Data Privacy**: Manufacturing data often contains proprietary information about production processes, defect patterns, and quality metrics that companies are reluctant to share.

2. **Data Silos**: Different factories or production lines operate independently, creating isolated data repositories that cannot be easily aggregated.

3. **Regulatory Constraints**: Industry regulations may restrict the transfer of quality control data across organizational boundaries.

**Federated Learning (FL)** offers a compelling solution to these challenges by enabling collaborative model training without centralizing sensitive data. In a federated setup, multiple participants (e.g., production lines or factories) train local models on their private data and share only model updates with a central server for aggregation.

## 1.2 Problem Statement

This project addresses the following research questions:

1. **Can federated learning achieve competitive anomaly detection performance compared to centralized approaches?**

2. **How does data heterogeneity (non-IID distributions) affect federated model performance in industrial anomaly detection?**

3. **What are the trade-offs between privacy preservation and model accuracy in federated industrial inspection systems?**

We investigate these questions using the **AutoVI (Automotive Visual Inspection) dataset**, a genuine industrial dataset from Renault Group containing images of automotive components with various defect types.

## 1.3 Contributions

This Stage 1 report presents the following contributions:

- **Data Loader Implementation**: Complete data loading pipeline for AutoVI v1.0.0 dataset with all 6 product categories (3,950 total images).

- **Baseline Model Development**: PatchCore implementation in progress, following unsupervised anomaly detection paradigm.

- **Federated Architecture Design**: Federated learning framework with 6 independent clients (one per product category) using memory bank aggregation.

- **Stage 1 Foundations**: Established infrastructure for local training without aggregation, ready for Stage 2 enhancements (privacy, fairness).
