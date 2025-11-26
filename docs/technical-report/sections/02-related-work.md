# 2. Related Work

## 2.1 Anomaly Detection in Industrial Settings

Unsupervised anomaly detection has gained significant attention in industrial quality control, where labeled defect data is scarce and defect types are diverse. Recent deep learning approaches can be categorized into:

**Reconstruction-based methods**: Autoencoders and GANs learn to reconstruct normal images; anomalies produce high reconstruction errors. While simple to implement, these methods often struggle with complex textures and subtle defects.

**Feature embedding methods**: Models like **PatchCore** [1] extract features from pre-trained CNNs and store representative normal patches in a memory bank. During inference, anomalies are detected as patches distant from the memory bank. PatchCore achieves state-of-the-art performance on benchmarks like MVTec AD.

**Knowledge distillation methods**: EfficientAD and similar approaches train student networks to mimic teacher networks on normal data; discrepancies indicate anomalies.

We adopt **PatchCore** for this project due to its strong performance, interpretability (anomaly maps correspond to patch distances), and natural adaptation to federated settings (memory banks can be aggregated).

## 2.2 Federated Learning

Federated Learning [2] enables distributed model training across multiple clients without sharing raw data. The canonical **FedAvg** algorithm aggregates client model updates using weighted averaging:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^t$$

where $w_k^t$ represents client $k$'s model weights and $n_k$ is the local dataset size.

Key challenges in FL include:
- **Non-IID data**: Performance degrades when client data distributions differ significantly [3].
- **Communication efficiency**: Transmitting large model updates is costly.
- **Privacy guarantees**: Additional mechanisms (differential privacy, secure aggregation) are needed for formal privacy.

## 2.3 Federated Anomaly Detection

Limited work has explored FL for anomaly detection. Most existing approaches focus on:
- Federated autoencoders for network intrusion detection
- Distributed sensor data anomaly detection
- Federated one-class classification

To our knowledge, **no prior work has applied federated learning to PatchCore-based industrial anomaly detection**. Our approach of aggregating memory banks (rather than neural network weights) presents unique opportunities for communication efficiency and privacy.

## References

[1] Roth, K., et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022.

[2] McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.

[3] Zhao, Y., et al. "Federated Learning with Non-IID Data." arXiv 2018.

[4] Carvalho, P., et al. "The Automotive Visual Inspection Dataset (AutoVI)." Zenodo 2024.

[5] Li, T., et al. "Federated Optimization in Heterogeneous Networks (FedProx)." MLSys 2020.
