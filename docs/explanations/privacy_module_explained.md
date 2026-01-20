# The Privacy Module: A Comprehensive Deep Dive

## First, Let's Understand WHY We Need This

Before diving into code, let's understand the **fundamental problem** this module solves.

---

## The Privacy Problem in Federated Learning

### The Naive Assumption
"We're not sharing raw images, just feature vectors. That's private, right?"

**Wrong!** Here's why:

```
ATTACK SCENARIO: Membership Inference Attack
────────────────────────────────────────────

Attacker has:
  - Access to the global memory bank (feature vectors)
  - A suspect image they want to check

Attack:
  1. Extract features from suspect image
  2. Check if these features are "unusually close" to memory bank entries
  3. If yes → That person's data was likely used for training!

This reveals: "Was Alice's factory data used in this model?"
This is a PRIVACY VIOLATION even without seeing raw images.
```

**Solution:** Add carefully calibrated noise to the features before sharing. This is **Differential Privacy (DP)**.

---

## The Core Concept: Differential Privacy

### The Intuitive Definition

Imagine two scenarios:
- **World A:** Alice's data IS in the training set
- **World B:** Alice's data is NOT in the training set

**Differential Privacy guarantees:** An attacker looking at the output (the model/memory bank) cannot reliably distinguish between World A and World B.

In other words: **Your participation doesn't significantly change the output.**

### The Mathematical Definition

A mechanism M is **(ε, δ)-differentially private** if for any two datasets D and D' that differ by one person's data, and for any output S:

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ
```

**Breaking this down:**
- `P[M(D) ∈ S]` = Probability of getting output S when using dataset D
- `e^ε` = The multiplicative privacy loss (e ≈ 2.718)
- `δ` = The probability that the guarantee completely fails

---

# Now Let's Understand Each Parameter

## 1. Epsilon (ε) - The Privacy Budget

### Definition
**Epsilon** measures how much information can leak about any individual. It's the "privacy loss" parameter.

### Intuitive Understanding

| Epsilon Value | Privacy Level | What It Means |
|---------------|---------------|---------------|
| ε = 0 | Perfect privacy | Output is independent of any individual (useless model) |
| ε = 0.1 | Very strong privacy | Nearly impossible to infer anything about individuals |
| ε = 1.0 | Strong privacy | Standard choice for sensitive data |
| ε = 5.0 | Moderate privacy | Some leakage, but still protected |
| ε = 10.0 | Weak privacy | Significant leakage possible |
| ε = ∞ | No privacy | Raw data effectively exposed |

### Why the Code Uses ε ∈ [0.1, 10.0]

```python
# From embedding_sanitizer.py, line 31-32
if not 0.1 <= self.epsilon <= 10.0:
    raise ValueError(f"Epsilon must be in [0.1, 10.0], got {self.epsilon}")
```

**Reasoning:**
- **Lower bound (0.1):** Below this, you need so much noise the model becomes useless
- **Upper bound (10.0):** Above this, privacy guarantees are too weak to be meaningful
- **Default (1.0):** Industry standard for "reasonably private"

### The Math Behind Epsilon

When ε = 1:
```
P[M(D) ∈ S] ≤ e^1 × P[M(D') ∈ S] + δ
            ≤ 2.718 × P[M(D') ∈ S] + δ
```

This means: The probability of any output changes by at most ~2.7× when adding/removing one person.

---

## 2. Delta (δ) - The Failure Probability

### Definition
**Delta** is the probability that the privacy guarantee *completely fails*. It's a "catastrophic failure" parameter.

### Intuitive Understanding

| Delta Value | What It Means |
|-------------|---------------|
| δ = 0 | Perfect guarantee (impossible with Gaussian noise) |
| δ = 1e-5 (0.00001) | Fails for 1 in 100,000 people |
| δ = 1e-7 (0.0000001) | Fails for 1 in 10 million people |
| δ = 1 | Guarantee always fails (useless) |

### Why the Code Uses δ ∈ (0, 1) with Default 1e-5

```python
# From embedding_sanitizer.py, line 25
delta: float = 1e-5
```

**Reasoning:**
- δ should be much smaller than 1/n where n = number of people in dataset
- For a dataset with 1,000 images from 100 factories, δ = 1e-5 means:
  - Probability of privacy failure for any single factory: 0.001%
- **Rule of thumb:** δ ≤ 1/(10 × n)

### Why Not δ = 0?

The **Gaussian mechanism** (adding normal noise) cannot achieve δ = 0. Here's why:

```
Gaussian distribution has infinite tails
─────────────────────────────────────────

               ┌────┐
            ┌──┤    ├──┐
         ┌──┤  │    │  ├──┐
      ───┤  │  │    │  │  ├───
         │  │  │    │  │  │    ← These tails go to infinity
      ───┘  └──┘    └──┘  └───

No matter how much noise you add, there's always a tiny
probability of producing ANY output value.
```

To achieve δ = 0, you'd need **Laplace noise** (which has bounded tails), but Gaussian is preferred for high-dimensional data like embeddings.

---

## 3. Sensitivity (Δf) - How Much Can One Person Change the Output?

### Definition
**Sensitivity** measures the maximum change in the output when one person's data is added or removed.

### Mathematical Definition (L2 Sensitivity)

```
Δf = max_{D, D'} ||f(D) - f(D')||₂
```

Where D and D' differ by exactly one person's data.

### In Our Context

When a client sends feature embeddings:
- **What's the function f?** Extracting patch features from images
- **What's the sensitivity?** How much can features change if we add/remove one image?

**The Problem:** Without bounds, sensitivity could be infinite!

```
Image A features: [0.1, 0.2, 0.3, ..., 0.5]     norm = 2.0
Image B features: [100, 200, 300, ..., 500]     norm = 1000.0

If we remove Image B, the output changes by up to 1000!
This is unbounded sensitivity → Cannot guarantee privacy.
```

**The Solution:** **Clipping** - force all embeddings to have bounded norm.

---

## 4. Clipping Norm - Bounding the Sensitivity

### Definition
**Clipping norm** is the maximum allowed L2 norm for any embedding vector. Embeddings exceeding this are scaled down.

### Why Clipping is Essential

```
WITHOUT CLIPPING:
─────────────────
Embedding norms: [1.2, 0.8, 1.5, 47.3, 0.9, 1.1]
                                 ↑
                            Outlier!

Sensitivity could be as high as 47.3!
Need HUGE noise to guarantee privacy → Useless model


WITH CLIPPING (norm = 1.0):
──────────────────────────
Before: [1.2, 0.8, 1.5, 47.3, 0.9, 1.1]
After:  [1.0, 0.8, 1.0,  1.0, 0.9, 1.0]
              ↑          ↑
         Already OK   Scaled down from 47.3 to 1.0

Sensitivity is now BOUNDED at 1.0
Can use reasonable noise → Useful model!
```

### The Clipping Formula

```python
# From embedding_sanitizer.py, lines 96-103
# Compute L2 norms
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Compute scaling factors (1.0 for norms <= clipping_norm)
scale = np.minimum(1.0, self.config.clipping_norm / (norms + 1e-8))

# Apply clipping
clipped = embeddings * scale
```

**Visual explanation:**

```
Original embedding: v = [3, 4]
L2 norm: ||v|| = √(3² + 4²) = √25 = 5

If clipping_norm = 1.0:
scale = min(1.0, 1.0/5) = 0.2

Clipped embedding: v' = 0.2 × [3, 4] = [0.6, 0.8]
New norm: ||v'|| = √(0.6² + 0.8²) = √1.0 = 1.0 ✓
```

**Geometrically:**

```
Before clipping:              After clipping:

     •  (far outside)              All points are
    / \                            within the ball
   /   •
  •     \                          ┌─────────┐
   \     •                         │  •  •   │
    \   /                          │ •    •  │
     \ /                           │    •    │
      • (inside ball)              └─────────┘
                                   radius = clipping_norm
```

### Why Default clipping_norm = 1.0?

```python
# From embedding_sanitizer.py, line 26
clipping_norm: float = 1.0
```

**Reasoning:**
1. **Normalized features:** After backbone feature extraction, embeddings are often normalized or near-normalized
2. **Balance:** Too small → clips too aggressively, loses information. Too large → needs more noise
3. **Convention:** 1.0 is a common choice that works well empirically
4. **Interpretability:** Easy to understand - "no embedding can exceed unit norm"

---

## 5. Sigma (σ) - The Noise Scale

### Definition
**Sigma** is the standard deviation of the Gaussian noise added to each embedding dimension.

### The Formula

```python
# From gaussian_mechanism.py, lines 57-66
def _compute_sigma(self) -> float:
    return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
```

**The mathematical formula:**

```
σ = Δf × √(2 × ln(1.25/δ)) / ε
```

Where:
- Δf = sensitivity (= clipping_norm in our case)
- δ = failure probability
- ε = privacy budget

### Why This Specific Formula?

This comes from the **Gaussian mechanism theorem** (Dwork & Roth, 2014):

> For a function f with L2 sensitivity Δf, adding Gaussian noise with σ = Δf × √(2 × ln(1.25/δ)) / ε achieves (ε, δ)-differential privacy.

**Derivation intuition:**
1. The Gaussian tail probability at distance t from mean is ~exp(-t²/2σ²)
2. We need this probability to relate to e^ε
3. Solving for σ gives the formula above

### Example Calculation

```
Given:
  ε = 1.0
  δ = 1e-5
  clipping_norm = 1.0 (= sensitivity)

σ = 1.0 × √(2 × ln(1.25 / 0.00001)) / 1.0
  = 1.0 × √(2 × ln(125000)) / 1.0
  = 1.0 × √(2 × 11.736) / 1.0
  = 1.0 × √23.472
  = 1.0 × 4.845
  = 4.845

So each embedding dimension gets noise ~ N(0, 4.845²)
```

### The Trade-off

```
Higher ε (less privacy) → Lower σ → Less noise → Better accuracy
Lower ε (more privacy)  → Higher σ → More noise → Worse accuracy

      Privacy ←─────────────────────────────→ Accuracy
         ε=0.1                                    ε=10
         σ=48.5                                   σ=0.48
      Very noisy                              Almost no noise
```

---

# Now Let's Walk Through Each File in Detail

---

# File 1: `gaussian_mechanism.py` - The Mathematical Core

## Purpose
This file implements the **Gaussian mechanism**, the fundamental building block for adding privacy-preserving noise.

## Class: `GaussianMechanism`

### Lines 10-21: Class Definition and Docstring

```python
class GaussianMechanism:
    """Gaussian mechanism for (epsilon, delta)-differential privacy.

    Adds calibrated Gaussian noise to achieve differential privacy guarantees.
    Uses the standard Gaussian mechanism formula from Dwork & Roth.

    Attributes:
        epsilon: Privacy parameter (lower = more private).
        delta: Failure probability.
        sensitivity: L2 sensitivity of the function (bounded by clipping).
        sigma: Computed noise scale.
    """
```

**Reference:** "Dwork & Roth" refers to:
> Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy"
>
> This is THE foundational textbook on differential privacy.

### Lines 23-55: `__init__` - Initialization

```python
def __init__(
    self,
    epsilon: float,
    delta: float,
    sensitivity: float,
):
    # Validate parameters
    if not 0.1 <= epsilon <= 10.0:
        raise ValueError(f"Epsilon must be in [0.1, 10.0], got {epsilon}")
    if not 0 < delta < 1:
        raise ValueError(f"Delta must be in (0, 1), got {delta}")
    if sensitivity <= 0:
        raise ValueError(f"Sensitivity must be positive, got {sensitivity}")

    self.epsilon = epsilon
    self.delta = delta
    self.sensitivity = sensitivity
    self.sigma = self._compute_sigma()  # ← Immediately compute noise scale
```

**What happens:**
1. **Validate inputs** - Ensure parameters are in valid ranges
2. **Store parameters** - Keep ε, δ, and sensitivity
3. **Compute σ immediately** - The noise scale is determined at initialization

### Lines 57-66: `_compute_sigma` - The Core Math

```python
def _compute_sigma(self) -> float:
    """Compute the noise scale sigma.

    Uses the standard Gaussian mechanism formula:
    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Returns:
        Noise scale sigma.
    """
    return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
```

**Let's trace through this step by step:**

```
Input: sensitivity=1.0, epsilon=1.0, delta=1e-5

Step 1: 1.25 / delta = 1.25 / 0.00001 = 125000

Step 2: ln(125000) = 11.736...

Step 3: 2 × 11.736 = 23.472

Step 4: √23.472 = 4.845

Step 5: sensitivity × 4.845 / epsilon = 1.0 × 4.845 / 1.0 = 4.845

Result: σ = 4.845
```

**Why 1.25 in the formula?**

This comes from careful analysis of the Gaussian tail. The factor 1.25 (instead of, say, 2.0 or 1.0) gives the tightest bound for the Gaussian mechanism. It's derived mathematically and is standard in DP literature.

### Lines 68-96: `add_noise` - Applying the Noise

```python
def add_noise(
    self,
    data: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add calibrated Gaussian noise to data."""
    # Create random generator with optional seed
    rng = np.random.default_rng(seed)

    # Generate Gaussian noise
    noise = rng.normal(loc=0.0, scale=self.sigma, size=data.shape)

    # Add noise and preserve dtype
    noised_data = data + noise.astype(data.dtype)

    return noised_data
```

**What happens:**

```
Input data: [0.5, -0.3, 0.8, 0.1]  (one embedding, 4 dimensions)
σ = 4.845

Generate noise ~ N(0, σ²) for each dimension:
noise = [2.1, -1.5, 0.3, -3.2]  (random samples)

Add noise:
noised = [0.5+2.1, -0.3-1.5, 0.8+0.3, 0.1-3.2]
       = [2.6, -1.8, 1.1, -3.1]
```

**Visual representation:**

```
Original embedding (clean signal):
─────────────────────────────────
[0.5, -0.3, 0.8, 0.1]

After adding Gaussian noise (σ=4.845):
─────────────────────────────────────
[2.6, -1.8, 1.1, -3.1]  ← Each value shifted by random amount

The noise is INDEPENDENT for each dimension!
This provides privacy while preserving the general structure.
```

**Why seed is important:**

```python
rng = np.random.default_rng(seed)
```

- With seed: Reproducible results (same noise every time)
- Without seed: Different noise each run
- In federated learning, each client uses a different seed (seed + client_id) for independent noise

---

# File 2: `embedding_sanitizer.py` - The Application Layer

## Purpose
This file applies the Gaussian mechanism specifically to **feature embeddings**. It handles the complete privacy workflow: clipping → noising.

## Class: `DPConfig` - Configuration Container

### Lines 12-36: The Configuration Dataclass

```python
@dataclass
class DPConfig:
    """Configuration for differential privacy.

    Attributes:
        enabled: Whether DP is enabled.
        epsilon: Privacy parameter (lower = more private).
        delta: Failure probability.
        clipping_norm: Maximum L2 norm for embeddings.
    """

    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    clipping_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration values."""
        if self.enabled:
            if not 0.1 <= self.epsilon <= 10.0:
                raise ValueError(f"Epsilon must be in [0.1, 10.0], got {self.epsilon}")
            if not 0 < self.delta < 1:
                raise ValueError(f"Delta must be in (0, 1), got {self.delta}")
            if self.clipping_norm <= 0:
                raise ValueError(f"Clipping norm must be positive, got {self.clipping_norm}")
```

**What is `@dataclass`?**

A Python decorator that automatically generates `__init__`, `__repr__`, and other methods. It's a clean way to define data containers.

**The `__post_init__` method:**

This runs AFTER the auto-generated `__init__`. It's used here for validation:
- Only validates if DP is enabled (no point validating if we're not using it)
- Ensures all values are in valid ranges

**Why these defaults?**

| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `enabled` | False | DP adds noise and reduces accuracy; opt-in only |
| `epsilon` | 1.0 | Standard "good privacy" choice |
| `delta` | 1e-5 | Small enough for datasets up to ~10,000 samples |
| `clipping_norm` | 1.0 | Works well with normalized features |

## Class: `EmbeddingSanitizer` - The Main Workhorse

### Lines 39-80: `__init__` - Setting Up

```python
class EmbeddingSanitizer:
    """Sanitizes embeddings using differential privacy.

    Applies L2 norm clipping followed by Gaussian noise addition
    to achieve (epsilon, delta)-differential privacy.
    """

    def __init__(self, config: DPConfig):
        self.config = config
        self.mechanism: Optional[GaussianMechanism] = None
        self._sanitization_count = 0
        self._stats: Dict = {
            "num_sanitizations": 0,
            "total_embeddings_processed": 0,
            "embeddings_clipped": 0,
            "avg_norm_before_clip": 0.0,
            "avg_norm_after_clip": 0.0,
        }

        if config.enabled:
            # Initialize Gaussian mechanism with clipping norm as sensitivity
            self.mechanism = GaussianMechanism(
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=config.clipping_norm,  # ← KEY INSIGHT!
            )
```

**The Critical Insight (line 73):**

```python
sensitivity=config.clipping_norm
```

**Why sensitivity = clipping_norm?**

After clipping, the maximum L2 norm of any embedding is `clipping_norm`. Therefore:
- If we add/remove one embedding, the maximum change in the aggregated result is at most `clipping_norm`
- This bounds the sensitivity!

```
Without clipping: sensitivity could be infinite
With clipping to C: sensitivity ≤ C

So we set sensitivity = clipping_norm
```

### Lines 82-115: `clip_embeddings` - Bounding Sensitivity

```python
def clip_embeddings(
    self, embeddings: np.ndarray
) -> Tuple[np.ndarray, int, float, float]:
    """Clip embeddings to maximum L2 norm.

    Projects embeddings with L2 norm > clipping_norm back to the
    ball of radius clipping_norm.
    """
    # Compute L2 norms
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute scaling factors (1.0 for norms <= clipping_norm)
    scale = np.minimum(1.0, self.config.clipping_norm / (norms + 1e-8))

    # Apply clipping
    clipped = embeddings * scale

    # Track statistics
    num_clipped = int(np.sum(norms.flatten() > self.config.clipping_norm))
    avg_norm_before = float(np.mean(norms))
    avg_norm_after = float(np.mean(np.linalg.norm(clipped, axis=1)))

    return clipped.astype(embeddings.dtype), num_clipped, avg_norm_before, avg_norm_after
```

**Let's trace through this step by step:**

```
Input embeddings (3 embeddings, 4 dimensions):
[
    [0.3, 0.4, 0.0, 0.0],    # norm = 0.5 (within bound)
    [0.6, 0.8, 0.0, 0.0],    # norm = 1.0 (exactly at bound)
    [3.0, 4.0, 0.0, 0.0],    # norm = 5.0 (exceeds bound!)
]
clipping_norm = 1.0

Step 1: Compute norms
norms = [[0.5], [1.0], [5.0]]

Step 2: Compute scale factors
scale = minimum(1.0, 1.0 / (norms + 1e-8))
      = minimum(1.0, [2.0, 1.0, 0.2])
      = [1.0, 1.0, 0.2]

Step 3: Apply scaling
clipped = embeddings * scale
        = [
            [0.3×1.0, 0.4×1.0, 0.0, 0.0],  = [0.3, 0.4, 0.0, 0.0]  unchanged
            [0.6×1.0, 0.8×1.0, 0.0, 0.0],  = [0.6, 0.8, 0.0, 0.0]  unchanged
            [3.0×0.2, 4.0×0.2, 0.0, 0.0],  = [0.6, 0.8, 0.0, 0.0]  scaled down!
          ]

Final norms: [0.5, 1.0, 1.0]  ← All within bound now!
```

**Why `+ 1e-8` in the denominator?**

```python
scale = np.minimum(1.0, self.config.clipping_norm / (norms + 1e-8))
```

This prevents division by zero if an embedding is all zeros:
```
If norms = 0:
  Without 1e-8: clipping_norm / 0 = infinity (crash!)
  With 1e-8:    clipping_norm / 1e-8 = very large number
                But min(1.0, very_large) = 1.0, so zero stays zero
```

### Lines 117-161: `sanitize` - The Complete Workflow

```python
def sanitize(
    self,
    embeddings: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sanitize embeddings with differential privacy.

    Applies L2 clipping followed by Gaussian noise addition.
    """
    if not self.config.enabled or self.mechanism is None:
        logger.debug("DP disabled, returning original embeddings")
        return embeddings

    logger.info(f"Sanitizing {len(embeddings)} embeddings with DP...")

    # Step 1: Clip embeddings to bound sensitivity
    clipped, num_clipped, avg_before, avg_after = self.clip_embeddings(embeddings)

    # Step 2: Add calibrated Gaussian noise
    sanitized = self.mechanism.add_noise(clipped, seed=seed)

    # Update statistics
    self._sanitization_count += 1
    self._stats["num_sanitizations"] = self._sanitization_count
    self._stats["total_embeddings_processed"] += len(embeddings)
    self._stats["embeddings_clipped"] += num_clipped
    # ... more stats ...

    return sanitized
```

**The Two-Step Sanitization Process:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SANITIZATION WORKFLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Raw embeddings from feature extraction
       [N embeddings × D dimensions]

                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: CLIPPING                                                             │
│                                                                              │
│ Purpose: Bound the sensitivity                                              │
│                                                                              │
│ Before: norms could be [0.5, 1.2, 47.3, 0.8, 2.1, ...]                      │
│ After:  norms are all ≤ clipping_norm                                       │
│                                                                              │
│ Math: v' = v × min(1, C/||v||)  where C = clipping_norm                     │
│                                                                              │
│ Effect: Some information loss, but sensitivity is now BOUNDED               │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: GAUSSIAN NOISE ADDITION                                              │
│                                                                              │
│ Purpose: Mask individual contributions                                       │
│                                                                              │
│ For each value v in the clipped embedding:                                  │
│   v' = v + noise,  where noise ~ N(0, σ²)                                   │
│                                                                              │
│ Effect: Individual values are obscured, but aggregate statistics preserved  │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼

OUTPUT: Sanitized embeddings
        - Individual embeddings are noisy
        - But aggregate (memory bank) is still useful!
        - And individual privacy is guaranteed
```

---

# File 3: `privacy_accountant.py` - Tracking the Budget

## Purpose
This file tracks how much privacy budget has been **spent** across multiple operations. Think of it as a "privacy bank account."

## Why Do We Need This?

### The Composition Problem

Each time you run a DP mechanism, you spend some privacy budget. If you run multiple times, the privacy loss **accumulates**:

```
Round 1: Spend ε₁ = 1.0
Round 2: Spend ε₂ = 1.0
Round 3: Spend ε₃ = 1.0

Total privacy loss ≤ ε₁ + ε₂ + ε₃ = 3.0  (basic composition)

After 10 rounds: Total ε = 10.0 → Weak privacy!
```

**The accountant tracks this accumulation** and warns if you exceed your budget.

## Class: `PrivacyExpenditure` - Recording One Spending Event

### Lines 9-16: The Data Record

```python
@dataclass
class PrivacyExpenditure:
    """Record of a single privacy expenditure."""

    epsilon: float      # How much ε was spent
    delta: float        # How much δ was spent
    round_num: int      # Which training round
    description: str = ""  # What operation (e.g., "Client 3 coreset sanitization")
```

This is simply a record of one "transaction" in the privacy bank.

## Class: `PrivacyAccountant` - The Bank Ledger

### Lines 19-47: `__init__` - Opening the Account

```python
class PrivacyAccountant:
    """Tracks cumulative privacy expenditure across multiple operations.

    Uses basic composition theorem for privacy accounting:
    - Total epsilon = sum of individual epsilons
    - Total delta = sum of individual deltas
    """

    def __init__(self, target_epsilon: Optional[float] = None):
        """Initialize the privacy accountant.

        Args:
            target_epsilon: Optional maximum privacy budget.
                If set, check_budget() will return False when exceeded.
        """
        self.target_epsilon = target_epsilon
        self.expenditures: List[PrivacyExpenditure] = []
```

**What is `target_epsilon`?**

This is your **privacy budget limit** - the maximum total ε you're willing to spend. Like setting a spending limit on a credit card:

```
target_epsilon = 5.0 means:
  "I'm willing to leak at most e^5 ≈ 148× probability ratio"

If total_epsilon exceeds 5.0, the accountant warns you!
```

### Lines 49-83: `record_expenditure` - Logging a Transaction

```python
def record_expenditure(
    self,
    epsilon: float,
    delta: float,
    round_num: int,
    description: str = "",
) -> None:
    """Record a privacy expenditure."""
    expenditure = PrivacyExpenditure(
        epsilon=epsilon,
        delta=delta,
        round_num=round_num,
        description=description,
    )
    self.expenditures.append(expenditure)

    total_eps, total_delta = self.get_total_privacy()
    logger.debug(
        f"Recorded privacy expenditure: round={round_num}, "
        f"epsilon={epsilon}, delta={delta}. "
        f"Total: epsilon={total_eps:.4f}, delta={total_delta:.2e}"
    )

    if self.target_epsilon and total_eps > self.target_epsilon:
        logger.warning(
            f"Privacy budget exceeded! Total epsilon={total_eps:.4f} > "
            f"target={self.target_epsilon}"
        )
```

**What happens:**
1. Create a record of this expenditure
2. Add it to the list
3. Compute running total
4. Check if budget exceeded → warn if so

### Lines 85-97: `get_total_privacy` - Basic Composition

```python
def get_total_privacy(self) -> Tuple[float, float]:
    """Get the total privacy spent using basic composition.

    Returns:
        Tuple of (total_epsilon, total_delta).
    """
    if not self.expenditures:
        return (0.0, 0.0)

    total_epsilon = sum(exp.epsilon for exp in self.expenditures)
    total_delta = sum(exp.delta for exp in self.expenditures)

    return (total_epsilon, total_delta)
```

**What is Basic Composition?**

This is the simplest way to track privacy across multiple operations:

```
Total ε = ε₁ + ε₂ + ε₃ + ... + εₙ
Total δ = δ₁ + δ₂ + δ₃ + ... + δₙ
```

**Theorem:** If you run n mechanisms with privacy (ε₁, δ₁), (ε₂, δ₂), ..., (εₙ, δₙ), the combined mechanism satisfies (Σεᵢ, Σδᵢ)-differential privacy.

**Note the comment on line 26-27:**

```python
# For tighter bounds, consider advanced composition or Rényi DP
# in future iterations.
```

**What is Advanced Composition?**

Basic composition is pessimistic. Advanced techniques give tighter bounds:

```
BASIC COMPOSITION:
  n operations with ε each → Total: n × ε

ADVANCED COMPOSITION:
  n operations with ε each → Total: O(√n × ε)  (much better!)

RÉNYI DP (ZCDP):
  Even tighter bounds, especially for Gaussian mechanism
```

This implementation uses basic composition for simplicity, but notes that future versions could use tighter accounting.

### Lines 99-121: Budget Checking

```python
def check_budget(self) -> bool:
    """Check if the privacy budget is still available."""
    if self.target_epsilon is None:
        return True

    total_epsilon, _ = self.get_total_privacy()
    return total_epsilon <= self.target_epsilon

def get_remaining_budget(self) -> Optional[float]:
    """Get the remaining epsilon budget."""
    if self.target_epsilon is None:
        return None

    total_epsilon, _ = self.get_total_privacy()
    return max(0.0, self.target_epsilon - total_epsilon)
```

**Usage example:**

```python
accountant = PrivacyAccountant(target_epsilon=5.0)

# After some operations...
accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=2)

print(accountant.check_budget())        # True (2.0 ≤ 5.0)
print(accountant.get_remaining_budget())  # 3.0

# After more operations...
accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=3)
accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=4)

print(accountant.check_budget())        # False (6.0 > 5.0)
print(accountant.get_remaining_budget())  # 0.0
```

### Lines 123-154: `get_report` - Comprehensive Reporting

```python
def get_report(self) -> Dict:
    """Get a comprehensive privacy report."""
    total_epsilon, total_delta = self.get_total_privacy()

    report = {
        "total_epsilon": total_epsilon,
        "total_delta": total_delta,
        "num_expenditures": len(self.expenditures),
        "target_epsilon": self.target_epsilon,
        "remaining_budget": self.get_remaining_budget(),
        "budget_exceeded": not self.check_budget() if self.target_epsilon else False,
        "expenditures_by_round": {},
    }

    # Group expenditures by round
    for exp in self.expenditures:
        round_key = f"round_{exp.round_num}"
        if round_key not in report["expenditures_by_round"]:
            report["expenditures_by_round"][round_key] = []
        report["expenditures_by_round"][round_key].append(...)

    return report
```

**Example report output:**

```json
{
  "total_epsilon": 3.0,
  "total_delta": 3e-5,
  "num_expenditures": 3,
  "target_epsilon": 5.0,
  "remaining_budget": 2.0,
  "budget_exceeded": false,
  "expenditures_by_round": {
    "round_1": [
      {"epsilon": 1.0, "delta": 1e-5, "description": "Client 0 coreset sanitization"},
      {"epsilon": 1.0, "delta": 1e-5, "description": "Client 1 coreset sanitization"}
    ],
    "round_2": [
      {"epsilon": 1.0, "delta": 1e-5, "description": "Client 0 coreset sanitization"}
    ]
  }
}
```

---

# How This All Connects to the Pipeline

## The Complete Privacy Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRIVACY IN THE FEDERATED PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

1. CONFIGURATION (federated_patchcore.py)
   ─────────────────────────────────────
   User sets privacy parameters:

   FederatedPatchCore(
       dp_enabled=True,
       dp_epsilon=1.0,      # Privacy budget per round
       dp_delta=1e-5,       # Failure probability
       dp_clipping_norm=1.0 # Maximum embedding norm
   )
                │
                ▼
2. CLIENT INITIALIZATION (client.py)
   ──────────────────────────────────
   Each client gets an EmbeddingSanitizer:

   if dp_config and dp_config.enabled:
       self.sanitizer = EmbeddingSanitizer(dp_config)
                │
                ▼
3. LOCAL TRAINING (client.py: build_local_coreset)
   ───────────────────────────────────────────────
   After building local coreset, apply DP:

   if self.sanitizer is not None:
       self.local_coreset = self.sanitizer.sanitize(
           self.local_coreset, seed=client_seed
       )

   Inside sanitize():
     a) Clip embeddings (bound sensitivity)
     b) Add Gaussian noise (achieve DP)
                │
                ▼
4. SERVER RECEIVES CORESETS (server.py)
   ────────────────────────────────────
   Server receives ALREADY SANITIZED coresets
   Records privacy expenditure:

   if self.privacy_accountant is not None:
       for stats in client_stats:
           self.privacy_accountant.record_expenditure(
               epsilon=stats["dp_epsilon"],
               delta=stats["dp_delta"],
               round_num=round_num
           )
                │
                ▼
5. PRIVACY REPORTING (after training)
   ─────────────────────────────────
   Get total privacy spent:

   report = server.get_privacy_report()
   # Returns total epsilon/delta across all clients and rounds
```

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRIVACY MODULE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────────┐
                            │      DPConfig       │
                            │                     │
                            │ enabled: bool       │
                            │ epsilon: float      │
                            │ delta: float        │
                            │ clipping_norm: float│
                            └──────────┬──────────┘
                                       │
                                       │ used by
                                       ▼
                            ┌─────────────────────┐
                            │ EmbeddingSanitizer  │
                            │                     │
                            │ - clip_embeddings() │───▶ Bound sensitivity
                            │ - sanitize()        │───▶ Full DP workflow
                            └──────────┬──────────┘
                                       │
                                       │ uses
                                       ▼
                            ┌─────────────────────┐
                            │  GaussianMechanism  │
                            │                     │
                            │ - _compute_sigma()  │───▶ Calibrate noise
                            │ - add_noise()       │───▶ Apply noise
                            └─────────────────────┘


                            ┌─────────────────────┐
                            │  PrivacyAccountant  │
                            │                     │
                            │ - record_expenditure│───▶ Log spending
                            │ - get_total_privacy │───▶ Sum up
                            │ - check_budget      │───▶ Warn if exceeded
                            │ - get_report        │───▶ Full summary
                            └─────────────────────┘
```

---

# Key Theoretical Concepts Summary Table

| Concept | Symbol | Definition | Typical Value | Effect |
|---------|--------|------------|---------------|--------|
| **Epsilon** | ε | Privacy loss parameter | 0.1 - 10.0 | Lower = more private, more noise |
| **Delta** | δ | Failure probability | 1e-5 to 1e-7 | Lower = stronger guarantee |
| **Sensitivity** | Δf | Max change from one person | = clipping_norm | Bounded by clipping |
| **Clipping Norm** | C | Max L2 norm of embeddings | 1.0 | Trade-off: info loss vs. noise |
| **Sigma** | σ | Noise standard deviation | ~4.85 for ε=1 | Computed from ε, δ, Δf |
| **Composition** | - | Combining multiple DP ops | Sum of ε's | Budget accumulates |

---

# Practical Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE FUNDAMENTAL TRADE-OFF                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    PRIVACY ◄────────────────────────────► ACCURACY

  Strong Privacy (ε = 0.1)                           Weak Privacy (ε = 10)
  ────────────────────────                           ─────────────────────
  • High noise (σ ≈ 48)                              • Low noise (σ ≈ 0.5)
  • Features heavily obscured                        • Features nearly intact
  • Poor anomaly detection                           • Good anomaly detection
  • Strong privacy guarantee                         • Weak privacy guarantee


         ε = 1.0 is the "sweet spot"
         ─────────────────────────────
         Reasonable privacy + reasonable accuracy
```

---

# References

1. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy" - The foundational textbook
2. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy" - DP-SGD paper
3. Mironov, I. (2017). "Rényi Differential Privacy" - Advanced composition

---

*This document explains the privacy module in the AutoVI Federated PatchCore project.*
