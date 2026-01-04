# Robustness Evaluation Summary

Generated: 2026-01-04T19:22:39.438956

## Key Findings

### Baseline (No Attack)

| Method | Mean Deviation |
|--------|----------------|
| Robust (Median) | 0.0098 |
| Standard (Mean) | 0.0024 |

### Attack Resistance

| Attack | Mal. % | Method | Mean Dev | TPR | FPR |
|--------|--------|--------|----------|-----|-----|
| noise | 10% | Robust+Scoring | 0.0111 | 1.00 | 0.00 |
| noise | 10% | Robust | 0.0111 | 0.00 | 0.00 |
| noise | 10% | Standard | 0.0199 | 0.00 | 0.00 |
| noise | 20% | Robust+Scoring | 0.0110 | 0.00 | 0.00 |
| noise | 20% | Robust | 0.0110 | 0.00 | 0.00 |
| noise | 20% | Standard | 0.0358 | 0.00 | 0.00 |
| noise | 30% | Robust+Scoring | 0.0121 | 0.00 | 0.00 |
| noise | 30% | Robust | 0.0121 | 0.00 | 0.00 |
| noise | 30% | Standard | 0.0205 | 0.00 | 0.00 |
| noise | 40% | Robust+Scoring | 0.0144 | 0.00 | 0.00 |
| noise | 40% | Robust | 0.0144 | 0.00 | 0.00 |
| noise | 40% | Standard | 0.0536 | 0.00 | 0.00 |
| scaling | 10% | Robust+Scoring | 0.0113 | 1.00 | 0.00 |
| scaling | 10% | Robust | 0.0113 | 0.00 | 0.00 |
| scaling | 10% | Standard | 0.1944 | 0.00 | 0.00 |
| scaling | 20% | Robust+Scoring | 0.0103 | 0.00 | 0.00 |
| scaling | 20% | Robust | 0.0103 | 0.00 | 0.00 |
| scaling | 20% | Standard | 0.3369 | 0.00 | 0.00 |
| scaling | 30% | Robust+Scoring | 0.0117 | 0.00 | 0.00 |
| scaling | 30% | Robust | 0.0117 | 0.00 | 0.00 |
| scaling | 30% | Standard | 0.1960 | 0.00 | 0.00 |
| scaling | 40% | Robust+Scoring | 0.0144 | 0.00 | 0.00 |
| scaling | 40% | Robust | 0.0144 | 0.00 | 0.00 |
| scaling | 40% | Standard | 0.4771 | 0.00 | 0.00 |
| sign_flip | 10% | Robust+Scoring | 0.0077 | 0.00 | 0.00 |
| sign_flip | 10% | Robust | 0.0077 | 0.00 | 0.00 |
| sign_flip | 10% | Standard | 0.0040 | 0.00 | 0.00 |
| sign_flip | 20% | Robust+Scoring | 0.0087 | 0.00 | 0.00 |
| sign_flip | 20% | Robust | 0.0087 | 0.00 | 0.00 |
| sign_flip | 20% | Standard | 0.0066 | 0.00 | 0.00 |
| sign_flip | 30% | Robust+Scoring | 0.0077 | 0.00 | 0.00 |
| sign_flip | 30% | Robust | 0.0077 | 0.00 | 0.00 |
| sign_flip | 30% | Standard | 0.0037 | 0.00 | 0.00 |
| sign_flip | 40% | Robust+Scoring | 0.0077 | 0.00 | 0.00 |
| sign_flip | 40% | Robust | 0.0077 | 0.00 | 0.00 |
| sign_flip | 40% | Standard | 0.0045 | 0.00 | 0.00 |

## Interpretation

- **Mean Deviation**: Lower is better (closer to honest data)
- **TPR (True Positive Rate)**: Higher is better (detects more malicious clients)
- **FPR (False Positive Rate)**: Lower is better (fewer false alarms)
