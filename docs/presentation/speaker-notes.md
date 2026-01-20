# Speaker Notes

## Slide 1: Title (30 seconds)

**Speaker: All (Member 1 leads)**

> "Good [morning/afternoon]. We are [Team Name], and today we'll present our Stage 1 work on Federated Learning for Industrial Anomaly Detection."

> "Our team consists of three members: [names and roles]. I'll hand off to different speakers throughout the presentation."

---

## Slide 2: Problem & Motivation (1.5 minutes)

**Speaker: Member 1**

> "Let's start with the problem we're trying to solve."

> "Industrial quality control generates massive amounts of visual inspection data. However, this data is often siloed across different factories or production lines."

[Point to diagram]

> "Factory A has its own inspection data, Factory B has its own, and so on. They can't simply pool their data together because of several constraints:"

> "First, **privacy concerns** - manufacturing data contains proprietary information about production processes and defect patterns."

> "Second, **regulatory restrictions** - industry regulations often prohibit sharing quality control data across organizational boundaries."

> "Third, **practical challenges** - different facilities handle different components, so the data is naturally heterogeneous."

> "This is where **Federated Learning** comes in. Instead of centralizing data, we train models locally at each site and only share model updates - in our case, feature representations - with a central server."

> "This preserves privacy while still enabling collaborative learning."

---

## Slide 3: AutoVI Dataset (1.5 minutes)

**Speaker: Member 1**

> "For our experiments, we use the AutoVI dataset - the Automotive Visual Inspection dataset from Renault Group."

[Point to sample images]

> "What makes this dataset special is that it's **genuine industrial data**, not synthetically generated. It contains images from real automotive production lines with authentic defects."

> "The dataset covers 6 types of automotive components: engine wiring harnesses, pipe clips, pipe staples, tank screws, underbody pipes, and underbody screws."

[Point to statistics table]

> "In total, we have about 1,500 training images and 2,400 test images. Importantly, the training set contains **only good images** - no defects. This is the unsupervised anomaly detection paradigm: the model learns what 'normal' looks like, and anything that deviates is flagged as anomalous."

> "The test set includes both good images and various defect types, with pixel-level ground truth annotations for evaluation."

> "I'll now hand off to [Member 2] to explain our model architecture."

---

## Slide 4: PatchCore Architecture (2 minutes)

**Speaker: Member 2**

> "For anomaly detection, we chose **PatchCore**, which is currently state-of-the-art on industrial benchmarks."

[Point to pipeline diagram]

> "The idea is elegantly simple. We take a pre-trained CNN - specifically WideResNet-50 - and use it to extract features from image patches."

> "During training, we only see normal images. We extract features from all patches and store the most representative ones in a **memory bank**."

> "The key innovation is **coreset selection** - instead of storing all features, we use a greedy algorithm to select a diverse subset that covers the feature space efficiently."

[Point to inference flow]

> "At inference time, for each patch in a test image, we compute its distance to the nearest neighbor in the memory bank. High distance means the patch is unlike anything we saw during training - it's anomalous."

> "Why did we choose PatchCore for federated learning?"

> "Three reasons: First, it achieves state-of-the-art performance. Second, the memory bank is a set of features that can be naturally aggregated across clients - unlike neural network weights that need gradient averaging. Third, the anomaly maps are interpretable - we can see exactly which regions triggered the detection."

---

## Slide 5: Federated Setup (2 minutes)

**Speaker: Member 2**

> "Now let me explain our federated architecture."

[Point to server-client diagram]

> "We simulate 5 federated clients, each representing a different production line in an automotive plant."

> "We tested two data partitioning strategies:"

> "First, **IID partitioning** - data is randomly and uniformly distributed. Each client sees 20% of each category. This is our baseline to understand the pure effect of federated learning."

> "Second, **Category-based partitioning** - this is more realistic. Client 1 only sees engine wiring. Client 2 handles underbody components. Client 3 does fasteners. And so on. This simulates real industrial scenarios where different facilities specialize in different parts."

> "The aggregation process is surprisingly simple. Each client extracts features from their local data and builds a local memory bank coreset. They send these coresets to the server. The server concatenates them and applies a global coreset selection to build the final memory bank."

> "A key advantage: we only need **one round** of communication. Unlike gradient-based federated learning that might need hundreds of rounds, our approach is extremely communication-efficient."

> "[Member 3] will now present our results."

---

## Slide 6: Baseline Results (1.5 minutes)

**Speaker: Member 3**

> "Let me start with our centralized baseline - this represents the upper bound performance when all data is pooled together."

[Point to bar chart]

> "We measure performance using AUC-sPRO at a 5% false positive rate. This metric evaluates how well the model localizes defects while keeping false positives low."

> "Across the 6 categories, we achieve a mean AUC-sPRO of 0.78. Performance varies by category - engine wiring performs best at 0.85, while underbody screw is most challenging at 0.70."

> "The variation reflects the inherent difficulty of each category. Engine wiring has well-defined defects and more training data. Underbody screw has class imbalance with very few anomalous test images."

> "For image-level classification, measured by AUC-ROC, we achieve a mean of 0.88."

> "These numbers align with what's reported in the PatchCore paper on similar industrial datasets, validating our implementation."

---

## Slide 7: Federated Comparison (1.5 minutes)

**Speaker: Member 3**

> "Now the key question: how does federated learning compare?"

[Point to comparison table]

> "With IID partitioning, federated PatchCore achieves 0.76 AUC-sPRO - only 3.2% below centralized. This is encouraging - we maintain 97% of performance while keeping data private."

> "However, with category-based partitioning - the realistic scenario - we see a larger gap. 0.71 AUC-sPRO represents a 9.4% drop from centralized."

[Point to highlight box]

> "This illustrates a fundamental trade-off. Memory bank aggregation is efficient, but when clients have very different data distributions, the global model may not adequately represent all categories."

> "The smaller clients suffer most. Pipe clip, with only 195 training images in one client, shows the largest degradation."

---

## Slide 8: Key Findings (1.5 minutes)

**Speaker: Member 3**

[Point to successes]

> "Let me summarize what we learned."

> "On the positive side: Federated PatchCore is feasible and practical. The single-round communication is a major advantage. And with IID data, performance is very close to centralized."

[Point to challenges]

> "The challenges center on data heterogeneity. Non-IID distributions cause significant performance drops. Clients with less data are underrepresented in the global model."

[Point to curves]

> "This FPR-sPRO curve visualization shows the gap more clearly. All methods converge at high FPR, but at strict thresholds - which matter most for quality control - the gap is substantial."

> "These limitations directly motivate our Stage 2 work."

---

## Slide 9: Stage 2 Roadmap (1 minute)

**Speaker: All (Member 1 leads)**

> "For Stage 2, we'll address two trust dimensions."

[Point to Privacy box]

> "First, **Privacy** through Differential Privacy. Currently, we share raw feature coresets. We'll add calibrated noise to provide formal privacy guarantees, tracking the privacy budget epsilon."

[Point to Fairness box]

> "Second, **Fairness**. We observed that smaller clients suffer more in the federated setting. We'll implement fairness-aware aggregation to ensure more balanced representation across categories."

> "We'll analyze the trade-offs: how much accuracy do we sacrifice for privacy? Can we reduce performance variance without hurting overall accuracy?"

> "Our deliverables will include a comprehensive 18-20 page report, the complete code repository, and a group presentation where each member presents their contribution."

---

## Slide 10: Q&A (2-3 minutes)

**Speaker: All**

> "Thank you for your attention. We're happy to take questions."

**Common responses:**

If asked about why PatchCore:
> "PatchCore's memory bank naturally fits federated aggregation - we can merge banks rather than averaging gradients. It also achieves state-of-the-art on industrial benchmarks."

If asked about only 5 clients:
> "The project guidelines specify 3-5 clients. Five is sufficient to demonstrate both IID and non-IID effects. The architecture scales to more clients."

If asked about privacy guarantees:
> "Stage 1 provides inherent privacy through local data retention. Stage 2 will add formal guarantees via Differential Privacy with quantified epsilon values."

If asked about real-world deployment:
> "Our simulation runs on a single machine. For production, you'd deploy actual client processes on factory servers with secure communication channels."
