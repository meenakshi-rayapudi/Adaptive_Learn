# AdaptiveLearn — Machine Learning Model Selection Rationale & EdTech AI Comparative Guide

## 1. Executive Summary
A central question in the design of **AdaptiveLearn** is:
> *Why are Gradient Boosted Decision Trees (XGBoost, LightGBM) and Random Forest chosen to model the learner, rather than Deep Learning (Neural Networks / Transformers), Classical Psychometrics (IRT / BKT), Collaborative Filtering, or pure LLM prompting?*

This document provides the theoretical, empirical, and architectural justification for this choice. It analyzes the mathematical properties of student behavioral data, surveys competing model families in Educational Data Mining (EDM), and specifies the **Model Tournament protocol** used to validate performance empirically.

---

## 2. The Nature of the Learner Feature Space

The learner model in AdaptiveLearn maps a student’s interaction history to a predicted **Knowledge Deficit Score** $D \in [0.0, 1.0]$ for each topic. The input feature vector is:

$$\vec{x} = \left[ Q, A, T, R, F, E, H \right]$$

Where:
* $Q \in [0.0, 1.0]$: Overall recent quiz performance (normalized percentage).
* $A \in [0.0, 1.0]$: Topic-specific accuracy across prior assessments.
* $T \in [500, 120000]$: Average response latency in milliseconds (measure of cognitive struggle).
* $R \in [0, 30]$: Elapsed days since last revision on this topic (spaced repetition signal).
* $F \in [0.0, 1.0]$: Flashcard mastery score derived from SM-2 ease and repetition intervals.
* $E \in [0, 500]$: Behavioral engagement count (remedial guides read, chat questions asked).
* $H \in [0, 10]$: Historical count of prior recommendations completed for this topic.

### Key Characteristics of this Feature Space:
1. **Heterogeneous Scales & Distributions:** $Q$ and $A$ are bounded ratios, $T$ is log-normally distributed milliseconds, $R$ is integer days, and $E$ is count data with extreme right-skew.
2. **Missing Modalities (Cold Start per Modality):** A student might regularly take quizzes but never review flashcards ($F = \text{NaN}$), or upload a new document without prior history on that topic ($H = 0$).
3. **Non-Linear Step Boundaries:** Cognitive forgetting and learning gaps operate along non-linear threshold steps rather than smooth linear curves (e.g., if $A < 40\%$ and $R > 5 \text{ days}$, deficit risk jumps discontinuously).

---

## 3. Why Tree-Based Models Outperform Deep Learning on Tabular Data

A common misconception in modern AI is that Deep Neural Networks are universally superior to tree-based algorithms. While Deep Learning dominates unstructured data (text, images, audio), **empirical machine learning research consistently proves that Gradient Boosted Decision Trees (GBDTs) outperform Deep Learning on tabular data**.

### Landmark Research Evidence:
In a seminal 2022 study published at NeurIPS:
> **Grinsztajn, Oyallon, & Varoquaux (NeurIPS 2022)** — *"Why do tree-based models still outperform deep learning on tabular data?"*  
> Benchmarking across 45 diverse tabular datasets, the authors demonstrated that tree-based models (XGBoost, LightGBM, CatBoost, Random Forest) significantly outperform Deep Neural Networks (MLPs, ResNet-like tabular architectures, and Transformer architectures like SAINT and FT-Transformer), while requiring orders of magnitude less tuning and compute.

### Why Trees Excel on AdaptiveLearn's Data:

| Dimensional Requirement | Deep Learning (MLP, Tabular Transformers) | Tree-Based GBDTs (XGBoost, LightGBM) |
| :--- | :--- | :--- |
| **Scale Invariance** | Highly sensitive. Requires extensive preprocessing (StandardScaler, Box-Cox, one-hot encoding). Unnormalized features cause gradient explosions or vanishing gradients. | **Fully invariant.** Decision trees split on ordering rank ($\le$ or $>$), rendering them completely immune to scale differences between milliseconds and percentages. |
| **Missing Value Handling** | Requires imputation (mean, median, KNN), which distorts the true distribution and introduces noise. | **Native branch routing.** XGBoost and LightGBM learn optimal default directions for $\text{NaN}$ values during node splitting. |
| **Sample Efficiency** | Requires tens of thousands of samples to learn meaningful representations without overfitting. | Trains effectively on modest sample sizes (hundreds to thousands of rows) while maintaining high generalization. |
| **Inference Latency & Cost** | Requires matrix multiplications, high RAM footprint, and often dedicated GPU acceleration. | **Sub-millisecond inference on CPU.** Model evaluation is simply traversing a set of binary conditional checks. |
| **Explainability (XAI)** | Black-box. Post-hoc attribution (Integrated Gradients, Saliency) is computationally expensive and approximate. | **Exact TreeSHAP.** Generates mathematically exact Shapley feature attributions in $O(TLD^2)$ time. |

---

## 4. Comprehensive Survey of EdTech & AI Model Families

To establish why alternative models were rejected or designated for future roadmap phases, we survey the five primary AI paradigms in education:

```
+-----------------------------------------------------------------------------------------------+
|                                    TAXONOMY OF EDTECH AI MODELS                               |
+-----------------------------------------------------------------------------------------------+
| Category               | Representative Models      | Primary Purpose in EdTech               |
+------------------------+----------------------------+-----------------------------------------+
| 1. Psychometrics       | IRT (Rasch, 2PL, 3PL), BKT | Static ability scoring on fixed tests   |
| 2. Collaborative Recs  | SVD, ALS, Neural CF (NCF)  | Peer-based item/course recommendations  |
| 3. Deep Sequence (KT)  | DKT (LSTM), SAINT (Transf) | Sequential step-by-step problem tracing |
| 4. Generative LLMs     | Llama 3.3, GPT-4, Claude   | Document synthesis & conversational RAG |
| 5. Tree Ensembles      | XGBoost, LightGBM, RF      | Multimodal behavioral learner modeling  |
+-----------------------------------------------------------------------------------------------+
```

---

### A. Psychometric Models: Item Response Theory (IRT) & Bayesian Knowledge Tracing (BKT)
* **Item Response Theory (IRT):** Models the probability of a correct answer as a logistic function of student latent ability $\theta$ and question parameters (difficulty $b$, discrimination $a$, guessing $c$):
  $$P(Y_{ij} = 1) = c_j + \frac{1 - c_j}{1 + e^{-a_j(\theta_i - b_j)}}$$
* **Bayesian Knowledge Tracing (BKT):** Uses a Hidden Markov Model (HMM) per skill to estimate the probability that a student has transitioned from unlearned to learned state, updated via Bayes' rule on each observation.
* **Why Not Primary for AdaptiveLearn:**
  * **Static & Uni-dimensional:** IRT assumes latent ability $\theta$ is fixed during the test session and only takes binary $(0, 1)$ correctness as input. It cannot incorporate response latency ($T$), time decay ($R$), flashcard mastery ($F$), or remedial engagement ($E$).
  * **Fixed Question Banks Required:** IRT requires thousands of student attempts per question to calibrate item difficulty $b_j$. In AdaptiveLearn, questions are dynamically generated from arbitrary uploaded documents, making pre-calibrated question parameters impossible.

---

### B. Collaborative Filtering & Matrix Factorization (Recommender Systems)
* **How It Works:** Popularized by Netflix and Spotify, algorithms like Singular Value Decomposition (SVD) or Neural Collaborative Filtering (NCF) recommend items based on peer similarity matrices: *"Students with similar taste/performance to you also benefited from Topic X."*
* **Why Not Primary for AdaptiveLearn:**
  * **Pedagogy $\neq$ Taste:** Academic learning is diagnostic and prerequisite-driven, not preference-driven. Recommending a topic because peers studied it ignores whether the student actually failed or mastered the prerequisites.
  * **Extreme Cold-Start Failure:** In a platform where students upload unique personal textbooks and notes, peer overlap is near zero. Collaborative filtering collapses when item interaction matrices are sparse.

---

### C. Deep Knowledge Tracing (DKT & Transformer SAINT)
* **How It Works:** Deep Knowledge Tracing (Piech et al., NeurIPS 2015) uses Recurrent Neural Networks (LSTM/GRU) or Self-Attention Transformers (SAINT, Choi et al., 2020) to ingest a student's full historical interaction sequence $[(q_1, a_1), (q_2, a_2), \dots, (q_t, a_t)]$ and predict success on question $q_{t+1}$.
* **Assessment for AdaptiveLearn:**
  * **Strengths:** Excellent sequential memory when trained on standardized math platforms (e.g., ASSISTments, EdNet) with a fixed taxonomy of hundreds of skills.
  * **Why It Is Slated for v3 (Not v1/v2):** DKT models require $10^5+$ historical sequences across a static curriculum to converge without severe overfitting. For a user uploading arbitrary new course notes, DKT has no pre-trained concept embeddings for the novel topics. GBDTs with aggregated feature vectors ($Q, A, T, R, F, E$) generalize far better across dynamic topics.

---

### D. Pure LLM-as-a-Recommender (Groq / Llama Prompting)
* **How It Works:** Passing the student's interaction log directly into an LLM prompt: *"Here is the student's study history. Recommend what topic they should study next and why."*
* **Why This Is an Anti-Pattern for the Core Learner Model:**
  1. **Latency & Cost:** An LLM inference takes 1.5–3.5 seconds and incurs ongoing API token charges on every page reload or button click. A tree-based model evaluates in **0.001 seconds** locally at zero cost.
  2. **Non-Determinism:** LLM generations are stochastic. If a student refreshes their dashboard without taking any quizzes, a prompt-based recommender might arbitrarily change its top recommendation, eroding user trust.
  3. **Hallucination of Metrics:** LLMs frequently invent historical facts (e.g., claiming *"You scored 20% on Deadlocks three days ago"* when the database records an 80% score from yesterday).
  4. **Lack of Scientific Evaluation:** You cannot calculate standardized ML loss functions (LogLoss, Brier score, ROC-AUC) or compute exact feature importance on raw LLM chat completions.

---

## 5. Comparative Evaluation Matrix Across All Paradigms

| Evaluation Criterion | Classical Psychometrics (IRT / BKT) | Collaborative Filtering (SVD / NCF) | Deep Knowledge Tracing (DKT / SAINT) | Pure LLM Prompting | Tree-Based GBDTs (XGBoost / LightGBM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Handles Heterogeneous Features ($Q, A, T, R, F, E$)** | ❌ Poor | ❌ Poor | ⚠️ Moderate | ⚠️ Textual only | ✅ **Superior** |
| **Resilience to Missing Modalities (NaN)** | ❌ Fails | ❌ Fails | ❌ Requires Imputation | ⚠️ Unpredictable | ✅ **Native** |
| **Works with Arbitrary Uploaded Documents** | ❌ No (Fixed bank needed) | ❌ No (Peer overlap needed) | ❌ No (Fixed skill graph) | ✅ Yes | ✅ **Yes** |
| **Inference Latency** | Fast (<5ms) | Fast (<10ms) | Moderate (50-200ms) | Slow (1500-3500ms) | **Ultra-Fast (<1ms)** |
| **Compute / Hardware Overhead** | Low (CPU) | Low (CPU) | High (GPU required) | Cloud API Dependent | **Low (CPU, 0 Cloud Cost)** |
| **Minimum Data Required to Train** | Large ($10^4+$ attempts) | Large ($10^4+$ interactions) | Massive ($10^5+$ sequences) | Zero-shot | **Small-to-Medium ($10^2 - 10^3$)** |
| **Mathematical Explainability (XAI)** | High (Parametric) | Low (Latent factors) | Very Low (Black-box) | Prone to Hallucination | **Exact (TreeSHAP)** |

---

## 6. The Model Tournament Protocol (Phase 3 Execution)

To ensure scientific rigor, we will not deploy XGBoost in isolation without proving its superiority on our data. During Phase 3, we will execute a formal **Model Tournament** comparing five distinct model architectures on the exact same dataset using identical chronological temporal splits.

```
                                  TOURNAMENT DATA PIPELINE
                                              |
                   +-----------------------------------------------------+
                   |   Time-Ordered Learner Logs (EdNet / ASSISTments)   |
                   +-----------------------------------------------------+
                                              |
                                     (Temporal 80/20 Split)
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
             +--------------------+                      +--------------------+
             |   TRAIN SET (T0)   |                      |  TEST SET (T_eval) |
             +--------------------+                      +--------------------+
                        |                                           |
    +-------------------+-------------------+                       |
    |                   |                   |                       |
    v                   v                   v                       |
[Logistic Reg]      [Random Forest]     [LightGBM]                  |
    |                   |                   |                       |
    v                   v                   v                       v
[MLP Neural Net]    [CatBoost]          [XGBoost] -----------> [EVALUATION BENCHMARK]
```

### Candidate Models in Tournament:
1. **Logistic Regression with L2 Regularization (Linear Baseline):** Tests if the relationship between features and deficit risk is simply linear.
2. **Multi-Layer Perceptron / MLP (Neural Baseline):** 3-layer fully connected network with ReLU activations and Dropout. Tests whether deep representations offer advantages.
3. **Random Forest (Bagging Baseline):** 200 estimators with bootstrap sampling. Highly resistant to noise and variance.
4. **LightGBM (Histogram-based GBDT):** Leaf-wise tree growth with gradient-based one-side sampling (GOSS). Optimized for speed and low memory.
5. **XGBoost (Exact/Approx GBDT):** Depth-wise tree growth with second-order gradient approximations and regularization ($\lambda, \gamma$).

### Tournament Evaluation Metrics:
* **ROC-AUC (Area Under ROC Curve):** Discriminative ability to distinguish between topics that will fail versus topics that will succeed.
* **Precision-Recall AUC (PR-AUC):** Vital for handling class imbalance (deficits are typically rarer than passes).
* **Brier Score:** Evaluates probability calibration (ensures a predicted 80% risk actually corresponds to an 80% failure rate).
* **Inference Latency (p95 milliseconds):** Evaluates user responsiveness under production conditions.
* **SHAP Stability:** Verifies that feature attributions are consistent and do not fluctuate wildly between adjacent runs.

---

## 7. Conclusion

The selection of **XGBoost and LightGBM with Random Forest** is grounded in:
1. **Mathematical alignment** with tabular, multi-scale, missing-value behavioral data.
2. **Pedagogical flexibility** across arbitrary user-uploaded PDFs.
3. **Operational efficiency** with zero-cost local CPU inference and exact TreeSHAP explainability.
4. **Empirical evidence** from top-tier machine learning research (NeurIPS) demonstrating tree-based superiority over neural networks for tabular problem formulations.
