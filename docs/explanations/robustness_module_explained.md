# The Robustness Module: A Comprehensive Deep Dive

## Understanding the Fundamental Problem: Byzantine Failures

Before we examine any code, we must first understand what problem this module solves and why it matters in the context of federated learning.

In a federated learning system, multiple clients collaboratively train a shared model. The central server relies on each client to honestly process their local data and send legitimate updates. However, this trust-based system has a critical vulnerability: what happens when some clients are not trustworthy?

The term "Byzantine" comes from the famous Byzantine Generals Problem in distributed computing, which describes a scenario where some participants in a system may fail or act maliciously in unpredictable ways. In our federated PatchCore context, a Byzantine client could be a compromised factory inspection station, a malicious insider, or simply a malfunctioning system that produces corrupted outputs.

The consequences of unaddressed Byzantine behavior can be severe. A single malicious client sending carefully crafted bad data could poison the entire global memory bank, causing the anomaly detection system to either miss real defects or flag good products as defective. Either outcome represents a significant failure in a manufacturing quality control system.

---

## The Threat Model: What Are We Defending Against?

To design effective defenses, we must first understand the attacks we face. In federated learning, Byzantine clients can pursue several malicious strategies.

The most straightforward attack involves sending random garbage data. A malfunctioning client might send corrupted feature vectors due to hardware failures, software bugs, or data corruption. While this seems simple, random noise can still significantly degrade model quality if not addressed.

A more sophisticated attack involves strategic manipulation. An intelligent adversary controlling one or more clients might craft their updates to maximally disrupt the learning process. They might try to make the global model perform poorly on specific categories of defects, effectively creating blind spots in the quality control system.

The attacks implemented in this module represent three canonical strategies that capture the essence of Byzantine behavior. The scaling attack amplifies legitimate updates by a large factor, attempting to dominate the aggregation process through sheer magnitude. The noise attack corrupts updates with random perturbations, injecting chaos into the learning process. The sign flip attack reverses the direction of updates entirely, attempting to make the model learn the opposite of what it should.

---

## The Defense Strategy: Robust Aggregation

The fundamental insight behind Byzantine-resilient aggregation is that honest clients, despite having different local data, will produce updates that are statistically similar to each other. Malicious clients, in contrast, must deviate from this norm to cause damage, and this deviation is detectable.

The defense strategy has two complementary components. First, robust aggregation methods replace the vulnerable averaging operation with statistical operations that naturally resist outliers. Second, anomaly detection identifies suspicious clients before their updates can influence the model.

---

## File 1: config.py — The Configuration Foundation

The configuration file establishes the parameters that control all robustness mechanisms. Understanding these parameters requires understanding both the mathematical foundations and practical considerations behind each choice.

### The RobustnessConfig Class

The configuration is implemented as a Python dataclass, which provides a clean way to define a collection of related parameters with default values and automatic validation.

```python
@dataclass
class RobustnessConfig:
    enabled: bool = False
    aggregation_method: str = "coordinate_median"
    num_byzantine: int = 0
    trim_fraction: float = 0.1
    client_scoring_method: str = "zscore"
    zscore_threshold: float = 3.0
```

Let us examine each parameter in depth.

### The enabled Parameter

This boolean flag determines whether any robustness mechanisms are active. The default value of False reflects an important design philosophy: robustness mechanisms add computational overhead and may slightly reduce model quality even when no attacks are present. Therefore, they should only be enabled when the threat model justifies their use.

In a trusted internal deployment where all clients are known and controlled, disabling robustness is reasonable. In a deployment involving external parties or untrusted networks, enabling robustness becomes essential.

### The aggregation_method Parameter

This parameter selects which robust aggregation algorithm to use. The implementation supports two methods, each with distinct mathematical properties.

The "coordinate_median" method computes the median value for each dimension across all client updates. The median has a remarkable property called the breakdown point, which measures what fraction of corrupted data an estimator can tolerate before becoming arbitrarily bad. The median has a breakdown point of 50%, meaning it remains accurate even when nearly half of the inputs are arbitrarily corrupted. This makes it ideal for Byzantine-resilient aggregation.

The "trimmed_mean" method discards a fraction of the most extreme values before computing the mean of the remainder. This approach offers a balance between the robustness of the median and the efficiency of the mean. The trim fraction determines how many extreme values to discard.

### The num_byzantine Parameter

This parameter represents our prior belief about how many clients might be malicious. While we cannot know this with certainty, having an estimate helps calibrate other parameters. For example, if we expect at most 2 Byzantine clients out of 10, we need less aggressive trimming than if we expect 4.

The validation ensures this value is non-negative, as negative counts are meaningless.

### The trim_fraction Parameter

When using trimmed mean aggregation, this parameter specifies what fraction of values to discard from each tail of the distribution. The value must be strictly between 0 and 0.5.

The lower bound of 0 is excluded because a trim fraction of zero would simply compute the ordinary mean, providing no robustness benefit. The upper bound of 0.5 is excluded because trimming 50% or more from each tail would leave nothing to average, or at best a single value.

The default value of 0.1 means we discard the lowest 10% and highest 10% of values before averaging. This provides protection against a moderate number of outliers while preserving most of the data's information content. The mathematical reasoning is that if we have n clients and expect at most f Byzantine ones, we should set trim_fraction slightly above f/n to ensure all malicious contributions are trimmed away.

### The client_scoring_method Parameter

This parameter selects the algorithm used to detect potentially malicious clients. The implementation supports "zscore" for statistical outlier detection or "none" to disable client scoring entirely.

The Z-score method is a statistical technique that measures how many standard deviations each client's behavior differs from the group average. Clients with unusually high Z-scores are flagged as potential outliers. This approach assumes that honest clients will produce similar updates, while malicious clients must deviate to cause harm.

### The zscore_threshold Parameter

This parameter sets the Z-score value above which a client is flagged as an outlier. The default value of 3.0 has deep statistical significance.

In a normal (Gaussian) distribution, approximately 99.7% of values fall within 3 standard deviations of the mean. This means that for a truly honest client whose behavior follows the normal distribution, there is only a 0.3% chance of being incorrectly flagged as an outlier. This low false positive rate is essential because incorrectly excluding honest clients degrades model quality.

The threshold of 3.0 represents a balance between sensitivity (catching actual malicious clients) and specificity (not falsely flagging honest clients). Lower thresholds catch more attacks but also produce more false alarms. Higher thresholds reduce false alarms but may miss subtle attacks.

The validation ensures the threshold is positive because a threshold of zero or below would be meaningless (every client would be flagged) or nonsensical.

### The Validation Logic

The __post_init__ method performs validation only when robustness is enabled, following the principle that we should not reject configurations for disabled features. This allows users to set up configurations in advance and enable them later without triggering validation errors.

---

## File 2: aggregators.py — The Heart of Byzantine Defense

The aggregators file implements the mathematical core of Byzantine resilience. The design uses object-oriented principles to create an extensible framework where new aggregation methods can be added without modifying existing code.

### The Abstract Base Class: RobustAggregator

The RobustAggregator class defines the interface that all robust aggregation methods must implement. This abstraction allows the rest of the system to work with any aggregation method without knowing its specific implementation.

```python
class RobustAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        client_updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
```

The method signature reveals important design decisions. The client_updates parameter is a list of numpy arrays, where each array contains one client's feature vectors. The shape of each array is [n_i, d], where n_i is the number of feature vectors from client i and d is the feature dimension (1536 in our PatchCore implementation). Different clients may contribute different numbers of vectors, which is why we use a list rather than a single tensor.

The optional weights parameter allows clients to be weighted by their data contribution. However, for truly robust aggregation methods like the median, weights are typically ignored because the median's robustness properties depend on treating all inputs equally.

The return type is a tuple containing the aggregated features and a dictionary of statistics. This allows each aggregation method to report relevant metrics about its operation, which is valuable for debugging and analysis.

### The Coordinate-Wise Median Aggregator

The CoordinateMedianAggregator class implements the primary robust aggregation method. Understanding its operation requires understanding both the mathematical properties of the median and the practical considerations of applying it to high-dimensional data.

#### The Mathematical Foundation

The median of a set of values is the middle value when the values are sorted. For an odd number of values, this is unique. For an even number, it is typically defined as the average of the two middle values.

The key property that makes the median useful for Byzantine resilience is its breakdown point of 50%. Formally, the breakdown point of an estimator is the largest fraction of arbitrary corruptions the estimator can tolerate while still returning a bounded result. For the median, you can replace up to half of the values with arbitrary values (including infinity) and the median will still be determined by the honest values.

To understand this intuitively, consider five values: [1, 2, 3, 100, 1000]. Despite the two extreme values, the median is 3, which accurately represents the "center" of the honest values [1, 2, 3]. The extreme values have no influence on the result.

In coordinate-wise application, we compute the median independently for each dimension of our feature vectors. If we have 5 clients each contributing 1536-dimensional vectors, we compute 1536 separate medians, one for each dimension.

#### The Sampling Strategy

The implementation faces a practical challenge: clients may contribute different numbers of feature vectors, but the median operation requires comparing corresponding vectors across clients. The solution is to sample a fixed number of vectors from each client.

```python
def __init__(self, num_samples_per_client: int = 100, seed: int = 42):
    self.num_samples_per_client = num_samples_per_client
    self.seed = seed
    self._rng = np.random.default_rng(seed)
```

The num_samples_per_client parameter determines how many vectors we sample from each client. The default of 100 provides a reasonable balance between computational efficiency and statistical representation. Sampling too few vectors might not capture the full distribution of each client's data. Sampling too many increases computation without proportional benefit.

The seed parameter ensures reproducibility. Using the same seed will produce the same sampling decisions, which is essential for debugging and scientific reproducibility.

#### The Aggregation Process

The aggregate method implements the full aggregation workflow. Let us trace through its logic step by step.

First, the method validates that we have at least one client with data to aggregate. An empty input list is meaningless and raises an appropriate error.

```python
if not client_updates:
    raise ValueError("No client updates provided")
```

Next, the method samples from each client's coreset. The sampling uses random selection, with replacement when a client has fewer vectors than requested and without replacement otherwise.

```python
if n_samples >= self.num_samples_per_client:
    indices = self._rng.choice(
        n_samples, size=self.num_samples_per_client, replace=False
    )
else:
    indices = self._rng.choice(
        n_samples, size=self.num_samples_per_client, replace=True
    )
```

Sampling without replacement means each vector can be selected at most once, which is appropriate when we have enough vectors. Sampling with replacement means the same vector can appear multiple times in our sample, which is necessary when a client has fewer vectors than our target sample size. This ensures all clients contribute equally to the aggregation regardless of their original data size.

After sampling, the method stacks all samples into a three-dimensional tensor with shape [num_clients, num_samples_per_client, feature_dim]. This organization aligns the data for the median computation.

```python
stacked = np.stack(sampled, axis=0)
```

Finally, the median is computed along the client axis (axis=0), producing a result with shape [num_samples_per_client, feature_dim].

```python
median_features = np.median(stacked, axis=0)
```

This operation independently computes, for each (sample_index, feature_dimension) pair, the median across all clients. The result is a robust aggregate that resists up to half of the clients being Byzantine.

---

## File 3: attacks.py — Understanding the Enemy

The attacks file implements attack simulations that allow us to test our robustness mechanisms. This might seem paradoxical—why would we implement attacks in our own system? The answer is that we cannot defend against threats we do not understand. By implementing realistic attacks, we can evaluate our defenses and ensure they work as intended.

The implementation explicitly notes these attacks are "for evaluation purposes only," emphasizing their role in testing rather than malicious use.

### The ModelPoisoningAttack Class

The class provides a unified interface for simulating different types of Byzantine attacks. The design allows easy configuration of attack parameters and application to specific clients.

```python
def __init__(
    self,
    attack_type: str = "scaling",
    scale_factor: float = 100.0,
    noise_std: float = 10.0,
    seed: Optional[int] = None,
):
```

#### The Scaling Attack

The scaling attack multiplies a client's updates by a large factor. The default scale_factor of 100.0 means malicious updates become 100 times larger than honest ones.

```python
def _scaling_attack(self, update: np.ndarray) -> np.ndarray:
    return update * self.scale_factor
```

The rationale behind this attack is that in simple averaging aggregation, the contribution of each client is proportional to the magnitude of their update. By scaling up their updates, a malicious client can dominate the aggregate, effectively overwriting the contributions of all honest clients.

Consider a simple example with 5 clients each contributing a single value. If honest clients contribute [1.0, 1.1, 0.9, 1.0, 1.0] and one malicious client scales their value by 100, we get [1.0, 1.1, 0.9, 100.0, 1.0]. The average becomes (1.0 + 1.1 + 0.9 + 100.0 + 1.0) / 5 = 20.8, which is dominated by the malicious contribution.

The coordinate-wise median defeats this attack because the malicious value, no matter how large, is simply an extreme value that gets ignored when computing the median. The median of [0.9, 1.0, 1.0, 1.0, 1.1, 100.0] remains around 1.0.

#### The Noise Attack

The noise attack adds random Gaussian noise to a client's updates. The noise_std parameter (default 10.0) controls the standard deviation of this noise.

```python
def _noise_attack(self, update: np.ndarray) -> np.ndarray:
    noise = self._rng.normal(0, self.noise_std, update.shape)
    return update + noise
```

This attack represents a Byzantine client that cannot craft targeted attacks but can inject chaos into the system. The noise destroys the structure of the feature vectors, making them useless for anomaly detection.

The standard deviation of 10.0 is chosen to be significantly larger than typical feature magnitudes (which are often around 1.0 after normalization), ensuring the noise dominates the signal.

#### The Sign Flip Attack

The sign flip attack reverses the sign of all values in a client's update.

```python
def _sign_flip_attack(self, update: np.ndarray) -> np.ndarray:
    return -update
```

This is perhaps the most insidious attack because the resulting values have the same magnitude as legitimate updates, making them harder to detect statistically. However, the reversed direction can cause the model to learn the opposite of what it should.

In the context of PatchCore, sign-flipped feature vectors would represent "opposite" visual patterns. When these are included in the memory bank, they corrupt the nearest-neighbor computations used for anomaly detection.

### The Apply Method

The apply method implements the attack workflow. It takes a list of all client updates and a list of indices identifying which clients are malicious.

```python
def apply(
    self,
    client_updates: List[np.ndarray],
    malicious_indices: List[int],
) -> List[np.ndarray]:
```

The method first creates copies of all updates to avoid modifying the originals. This is important for fair evaluation—we want to compare the poisoned updates against the original honest updates.

```python
result = [u.copy() for u in client_updates]
```

Then it applies the configured attack to each malicious client's updates.

```python
for idx in malicious_indices:
    if self.attack_type == "scaling":
        result[idx] = self._scaling_attack(result[idx])
    elif self.attack_type == "noise":
        result[idx] = self._noise_attack(result[idx])
    elif self.attack_type == "sign_flip":
        result[idx] = self._sign_flip_attack(result[idx])
```

The result is a list of updates where honest clients' data is unchanged and malicious clients' data has been corrupted according to the attack type.

---

## File 4: client_scoring.py — Detecting the Adversary

While robust aggregation provides passive defense by limiting the impact of malicious updates, client scoring provides active defense by identifying and potentially excluding suspicious clients. This complementary approach strengthens the overall security posture.

### The ClientScore Data Class

The ClientScore class encapsulates all information about a client's anomaly assessment.

```python
@dataclass
class ClientScore:
    client_id: int
    score: float
    is_outlier: bool
    details: Dict[str, Any] = field(default_factory=dict)
```

The client_id identifies which client this score applies to. The score is a numerical measure of how anomalous the client's behavior appears, with higher values indicating more suspicious behavior. The is_outlier boolean provides a binary classification based on comparing the score to a threshold. The details dictionary contains additional metrics that explain why the client received its score.

### The ZScoreDetector Class

The Z-score detector implements statistical outlier detection based on the assumption that honest clients will behave similarly to each other, while malicious clients must deviate to cause harm.

#### The Statistical Foundation of Z-Scores

The Z-score, also called the standard score, measures how many standard deviations a value is from the mean of its distribution. Mathematically, for a value x in a distribution with mean μ and standard deviation σ, the Z-score is:

```
Z = (x - μ) / σ
```

A Z-score of 0 means the value equals the mean. A Z-score of 1 means the value is one standard deviation above the mean. A Z-score of -2 means the value is two standard deviations below the mean.

The key insight is that in a normal distribution, Z-scores directly correspond to probabilities. A Z-score greater than 3 in absolute value occurs with probability less than 0.3% (roughly 1 in 370). Therefore, if a client's behavior yields a Z-score above 3, it is very unlikely to be a typical honest client.

#### The Detection Process

The score_clients method implements the full detection workflow. Let us trace through its operation.

First, the method computes summary statistics for each client's feature vectors.

```python
norms = np.linalg.norm(update, axis=1)
client_stats.append({
    "mean_norm": float(np.mean(norms)),
    "std_norm": float(np.std(norms)),
    "max_norm": float(np.max(norms)),
})
```

The choice of statistics is significant. We compute the L2 norm of each feature vector, then summarize these norms with their mean, standard deviation, and maximum. These three metrics capture different aspects of a client's behavior.

The mean norm captures the typical magnitude of the client's feature vectors. A malicious client using a scaling attack would have an unusually high mean norm.

The standard deviation of norms captures how variable the client's feature vectors are. A malicious client using a noise attack would have an unusually high standard deviation because random noise increases variability.

The maximum norm captures the most extreme feature vector. A malicious client might have a few extremely large vectors that stand out even if their mean appears normal.

Next, the method computes Z-scores for each metric across all clients.

```python
for name, values in [
    ("mean_norm", mean_norms),
    ("std_norm", std_norms),
    ("max_norm", max_norms),
]:
    mean = np.mean(values)
    std = np.std(values)
    if std > 1e-10:
        z_scores[name] = np.abs((values - mean) / std)
    else:
        z_scores[name] = np.zeros_like(values)
```

Note the absolute value applied to Z-scores. We care about deviation in either direction—both unusually high and unusually low values are suspicious. The check for standard deviation greater than 1e-10 prevents division by zero when all clients have identical statistics (which would indicate very similar or identical data).

Finally, the method classifies each client based on their maximum Z-score across all metrics.

```python
max_z = max(
    z_scores["mean_norm"][i],
    z_scores["std_norm"][i],
    z_scores["max_norm"][i],
)
is_outlier = max_z >= self.threshold
```

Using the maximum across metrics implements an "any of" logic: a client is flagged if they are anomalous on any metric. This is appropriate because different attacks produce different statistical signatures, and we want to catch them all.

---

## How Robustness Integrates with the Pipeline

Understanding how these components work together requires seeing how they integrate into the federated training process. The integration happens primarily in the server component, which orchestrates the aggregation of client updates.

### Configuration at System Initialization

When creating a FederatedPatchCore system, robustness parameters are passed to the constructor and used to configure the robustness components.

```python
self.robustness_config = RobustnessConfig(
    enabled=robustness_enabled,
    aggregation_method=robustness_aggregation,
    client_scoring_method="zscore" if robustness_enabled else "none",
    zscore_threshold=robustness_zscore_threshold,
)
```

The configuration flows to the server, which creates the appropriate aggregator and detector.

```python
if robustness_config and robustness_config.enabled:
    if robustness_config.aggregation_method == "coordinate_median":
        self.robust_aggregator = CoordinateMedianAggregator(
            num_samples_per_client=global_bank_size // 10
        )
    if robustness_config.client_scoring_method == "zscore":
        self.client_scorer = ZScoreDetector(
            threshold=robustness_config.zscore_threshold
        )
```

### Attack Simulation During Training

If attack simulation is enabled for evaluation purposes, attacks are applied after clients submit their coresets but before aggregation.

```python
if self.attack is not None and self.malicious_indices:
    client_coresets = self.attack.apply(client_coresets, self.malicious_indices)
```

This simulates the scenario where malicious clients have corrupted their updates before sending them to the server. The server receives a mix of honest and corrupted coresets.

### Robust Aggregation at the Server

When the server aggregates client coresets, it uses the robust aggregator if one is configured.

```python
if self.robust_aggregator is not None:
    self.global_features, robust_stats = self._robust_aggregate(
        self._pending_coresets, seed=seed
    )
```

The _robust_aggregate method first scores clients to detect outliers, then applies robust aggregation.

```python
def _robust_aggregate(self, client_coresets, seed=42):
    stats = {"robustness_enabled": True}

    if self.client_scorer:
        scores = self.client_scorer.score_clients(client_coresets)
        stats["num_outliers"] = sum(1 for s in scores if s.is_outlier)

    result, agg_stats = self.robust_aggregator.aggregate(client_coresets, weights=None)
    return result, stats
```

The client scoring results are recorded in the statistics for analysis but do not currently exclude clients from aggregation. The robust aggregator's statistical properties naturally limit the influence of outliers without requiring explicit exclusion.

---

## The Mathematics of Byzantine Fault Tolerance

To fully appreciate why these mechanisms work, we need to understand the mathematical principles underlying Byzantine fault tolerance.

### The Breakdown Point

The breakdown point is a measure of robustness that quantifies the maximum fraction of arbitrary corruptions an estimator can handle. Formally, for an estimator T applied to a dataset of n points, the breakdown point is the largest fraction m/n such that replacing any m points with arbitrary values keeps T bounded.

For the arithmetic mean, the breakdown point is 0%. A single corrupted point set to infinity makes the mean infinite. This is why simple averaging fails catastrophically under Byzantine attacks.

For the median, the breakdown point is 50%. Replacing up to half the values with arbitrary values does not change which value is in the middle of the sorted order (it is still determined by the honest majority). This is why the median provides strong Byzantine resilience.

For the trimmed mean with trim fraction α, the breakdown point is α. If we trim α from each tail before averaging, we can tolerate up to α fraction of corrupted values. The default α = 0.1 gives a breakdown point of 10%.

### The Trade-off Between Robustness and Efficiency

Robustness comes at a cost. The mean is the minimum variance unbiased estimator under Gaussian assumptions—when data is clean, no estimator is more efficient. The median, while robust, has higher variance for clean data.

For clean Gaussian data, the relative efficiency of the median compared to the mean is only about 64%. This means you would need about 56% more data to achieve the same accuracy with the median as with the mean.

In practice, this trade-off is acceptable when Byzantine attacks are a genuine threat. A 36% efficiency loss is preferable to complete model corruption.

### Why Z-Score of 3 is the Threshold

The threshold of 3 for Z-score detection comes from the properties of the normal distribution and a balancing of error types.

In a standard normal distribution, the probability of |Z| ≥ 3 is approximately 0.0027, or about 0.27%. This is the false positive rate—the probability that an honest client behaving normally would be incorrectly flagged as malicious.

For a system with 5 honest clients operating over many rounds, a false positive rate of 0.27% means false alarms are rare but not impossible. This is generally acceptable because robust aggregation can still handle occasional false positives.

Setting the threshold lower (say, Z = 2) would increase the false positive rate to about 4.5%, which would cause many honest clients to be incorrectly flagged. Setting it higher (say, Z = 4) would reduce the false positive rate but also reduce the ability to detect actual attacks.

The threshold of 3 represents decades of statistical practice in outlier detection and provides a well-tested balance between sensitivity and specificity.

---

## Visual Summary of the Robustness Module

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ROBUSTNESS MODULE ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │  RobustnessConfig   │
                              │                     │
                              │  enabled            │
                              │  aggregation_method │
                              │  num_byzantine      │
                              │  trim_fraction      │
                              │  client_scoring     │
                              │  zscore_threshold   │
                              └──────────┬──────────┘
                                         │
                                         │ configures
                          ┌──────────────┴──────────────┐
                          │                             │
                          ▼                             ▼
           ┌──────────────────────┐      ┌──────────────────────┐
           │    ZScoreDetector    │      │ CoordinateMedian     │
           │                      │      │    Aggregator        │
           │  Detects suspicious  │      │                      │
           │  clients based on    │      │  Aggregates using    │
           │  statistical         │      │  coordinate-wise     │
           │  deviation           │      │  median (robust to   │
           │                      │      │  50% Byzantine)      │
           └──────────┬───────────┘      └──────────┬───────────┘
                      │                             │
                      │                             │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Robust Aggregation   │
                        │                        │
                        │  1. Score all clients  │
                        │  2. Flag outliers      │
                        │  3. Apply robust       │
                        │     aggregation        │
                        │  4. Return global      │
                        │     memory bank        │
                        └────────────────────────┘


              ┌──────────────────────────────────────────┐
              │         ModelPoisoningAttack             │
              │           (For Testing Only)             │
              │                                          │
              │  Simulates malicious client behavior:    │
              │                                          │
              │  • Scaling: Multiply by large factor     │
              │  • Noise: Add random perturbations       │
              │  • Sign Flip: Reverse all values         │
              │                                          │
              │  Used to evaluate defense effectiveness  │
              └──────────────────────────────────────────┘
```

---

## The Complete Flow from Attack to Defense

To understand how all pieces work together, let us trace through a complete scenario where an attacker attempts to poison the federated learning process.

### The Attack Scenario

Imagine a federated PatchCore system with 5 clients. One client (client index 0) has been compromised and will attempt a scaling attack with scale_factor = 100.

During training, each client extracts features from their local data and builds a local coreset. The honest clients produce coresets with feature vectors having typical norms around 1.0. The malicious client multiplies their coreset by 100, producing vectors with norms around 100.0.

### Without Robustness

If robustness is disabled, the server uses standard aggregation (federated_coreset strategy). This strategy concatenates all coresets and applies global coreset selection.

The problem is that the global coreset selection uses greedy k-center, which iteratively selects points that are far from already-selected points. The malicious client's vectors, being 100 times larger, occupy a different region of feature space and will likely all be selected as "diverse" points. The resulting global memory bank is dominated by the malicious contributions.

When this corrupted memory bank is used for anomaly detection, the distance computations are distorted. Normal test images might be flagged as anomalies because they are far from the malicious vectors. Actual anomalies might be missed because the malicious vectors create spurious "normal" patterns.

### With Robustness

When robustness is enabled with coordinate_median aggregation and zscore client scoring, the defense proceeds in two stages.

First, the Z-score detector analyzes each client's coreset. It computes the mean norm for each client: [100.0, 1.0, 1.1, 0.9, 1.0]. The mean of these values is about 20.6, but this is heavily influenced by the outlier. The standard deviation is about 44.1.

The Z-score for client 0 is |100.0 - 20.6| / 44.1 ≈ 1.8, which is below the threshold of 3. Interestingly, this shows that a single outlier can distort even the detection process when it is extreme enough to affect the mean and standard deviation.

However, the coordinate-wise median aggregation provides robust protection regardless. The aggregator samples vectors from each client and computes the median across clients for each dimension. Even though client 0's vectors are 100 times larger, they represent only 1 out of 5 values for each dimension. The median naturally selects from the honest majority.

The resulting global memory bank contains vectors with typical norms around 1.0, completely unaffected by the malicious contributions. Anomaly detection proceeds correctly on clean data.

---

## Key Theoretical Concepts Summary

The Byzantine Generals Problem describes a scenario in distributed computing where some participants may fail or behave maliciously in arbitrary ways. The name comes from a thought experiment about coordinating generals who must agree on a battle plan but cannot trust all messengers. In federated learning, clients are analogous to generals, and their model updates are the messages.

The Breakdown Point of an estimator measures its robustness as the maximum fraction of corrupted data it can tolerate while remaining bounded. The mean has a breakdown point of 0% (completely vulnerable), while the median has a breakdown point of 50% (maximally robust for symmetric estimators).

The Z-Score standardizes a value by expressing it as the number of standard deviations from the mean. Values with |Z| > 3 occur with less than 0.3% probability under normal distribution assumptions, making them likely outliers.

Coordinate-Wise Median applies the median operation independently to each dimension of high-dimensional data. This preserves the robustness properties of the median while handling vectors rather than scalars.

Trimmed Mean discards a fraction of the most extreme values before computing the mean. With trim fraction α, it has breakdown point α and provides a balance between robustness and efficiency.

---

## Practical Recommendations

When deploying the robustness module, several considerations guide the configuration choices.

For systems where all clients are trusted and controlled, disabling robustness avoids unnecessary computational overhead and maintains optimal model quality.

For systems involving untrusted networks or external parties, enabling robustness with coordinate_median aggregation provides strong protection against up to 50% malicious clients. This is appropriate when the threat model includes potentially compromised clients.

The zscore_threshold of 3.0 is appropriate for most scenarios. Lower thresholds (2.0-2.5) can be used when higher sensitivity is desired at the cost of more false positives. Higher thresholds (3.5-4.0) reduce false positives but may miss subtle attacks.

For testing and evaluation, the attack simulation capabilities allow systematic assessment of defense effectiveness. Running experiments with different attack types and malicious fractions reveals the boundaries of the system's resilience.

---

## References

The theoretical foundations of this module draw from established research in distributed computing and robust statistics.

Lamport, L., Shostak, R., & Pease, M. (1982). "The Byzantine Generals Problem." ACM Transactions on Programming Languages and Systems.

Huber, P. J. (1981). "Robust Statistics." Wiley Series in Probability and Statistics.

Blanchard, P., et al. (2017). "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." NeurIPS.

Yin, D., et al. (2018). "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates." ICML.

---

*This document explains the robustness module in the AutoVI Federated PatchCore project.*
