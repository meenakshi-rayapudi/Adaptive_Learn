# AdaptiveLearn — Architectural Decisions, Alternatives & Dataset Strategy

## Executive Summary
This document provides an exhaustive Architectural Decision Record (ADR) and Public Educational Dataset Strategy for **AdaptiveLearn**. It evaluates all critical design dimensions, assesses all competing architectural alternatives, justifies why the chosen path is technically and pedagogically superior, and outlines how to leverage publicly available student interaction datasets to train and bootstrap the Machine Learning models.

---

# Part 1: Curated Public Student Datasets for Model Training

To train and validate the learner-modeling engine before accumulating thousands of proprietary user interactions, AdaptiveLearn can leverage industry-standard benchmark datasets from Educational Data Mining (EDM) and Intelligent Tutoring Systems (ITS).

### 1. EdNet (by Riiid) — *Best for General Correctness & Response Time*
* **Source:** [Kaggle / Riiid AIEd Challenge](https://www.kaggle.com/c/riiid-test-answer-prediction) & [EdNet GitHub](https://github.com/riiid/ednet)
* **Scale:** Over 130 million student interactions from 780,000+ students on the Santa AI tutoring platform.
* **Key Fields Available:**
  * `user_id`: Unique student identifier.
  * `content_id`: Question or lecture item ID.
  * `task_container_id`: Batch or session container of questions.
  * `user_answer`: Selected answer choice.
  * `answered_correctly`: Target outcome ($0$ or $1$).
  * `prior_question_elapsed_time`: Response time in milliseconds (measures latency/struggle).
  * `prior_question_had_explanation`: Whether student viewed remedial explanations ($E$).
  * `tags`: Hierarchical Knowledge Component (KC) / Skill identifiers ($topic\_id$).
* **Mapping to AdaptiveLearn:**
  * Maps 1:1 to Quiz Performance ($Q$), Topic-Wise Accuracy ($A$), Time Spent ($T$), and Remedial Engagement ($E$).
  * Can be directly used to train an XGBoost or LightGBM model to predict whether a student will fail a specific skill on their next encounter given their historical lag, prior accuracy, and attempt count.

---

### 2. ASSISTments Benchmark Datasets (2009–2010, 2012–2013, 2017) — *Gold Standard for Knowledge Tracing*
* **Source:** [Worcester Polytechnic Institute (WPI) ASSISTments Data Mining Repository](https://sites.google.com/site/assistmentsdata/)
* **Scale:** Multi-year longitudinal student problem-solving records across thousands of middle/high school students (~300k to ~3M rows per dataset).
* **Key Fields Available:**
  * `user_id`, `problem_id`, `skill_id` (or `skill_name`).
  * `correct`: First-attempt correctness.
  * `attempt_count`: Number of attempts required to solve.
  * `ms_first_response`: Response latency in milliseconds.
  * `hint_count`: Number of hints requested during the problem.
  * `opportunity`: The cumulative count of times this student has encountered this specific skill (crucial for practice frequency $R$).
* **Mapping to AdaptiveLearn:**
  * The `opportunity` and `skill_id` structure matches AdaptiveLearn's per-topic cumulative history.
  * Excellent for learning the relationship between practice frequency ($R$), hints/remedial needs, and eventual topic mastery.

---

### 3. Duolingo Spaced Repetition / Halflife Regression Dataset (Settles & Meeder) — *Best for Flashcard Mastery ($F$) & Spaced Intervals ($R$)*
* **Source:** [Harvard Dataverse (Settles & Meeder, ACL 2016)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME)
* **Scale:** 13 million active recall trials from 115,000 language learners across 40 days.
* **Key Fields Available:**
  * `user_id`: Learner identifier.
  * `lexeme_id`: Concept / word item identifier.
  * `history_seen`: Cumulative times this card has been reviewed.
  * `history_correct`: Cumulative times recalled successfully.
  * `time_delta`: Exact elapsed time in seconds since the student last reviewed this item.
  * `p_recall`: Empirical probability of recall.
* **Mapping to AdaptiveLearn:**
  * Maps directly to the Flashcard & Spaced Repetition engine.
  * Used to train or calibrate the forgetting curve $R(t) = e^{-\frac{t}{S}}$ and evaluate the SM-2 algorithm, ensuring flashcard mastery scores ($F$) reflect true retention decay over days.

---

### 4. OULAD (Open University Learning Analytics Dataset) — *Best for Multi-Session Engagement ($E$) & Drop-Off*
* **Source:** [Open University Learning Analytics](https://analyse.kmi.open.ac.uk/open_dataset) / Kaggle
* **Scale:** 32,593 students across 22 courses, with 10.6 million interaction logs with the Virtual Learning Environment (VLE).
* **Key Fields Available:**
  * `student_id`, `code_module`, `code_presentation`.
  * `date`: Interaction day relative to course start.
  * `sum_click`: Number of clicks per day on course resources (readings, quizzes, forums).
  * `assessment_type`, `score`, `date_submitted`.
* **Mapping to AdaptiveLearn:**
  * Bridges session duration ($T$) and broad interaction counts ($E$) with overall academic performance ($Q$).

---

### 5. Strategy: How to Use External Datasets in AdaptiveLearn
Because public datasets use fixed subjects (e.g., TOEIC English in EdNet, Math in ASSISTments, Languages in Duolingo) while AdaptiveLearn ingests arbitrary user PDFs, the datasets should be used through a **Feature-Aligned Transfer Pipeline**:
1. **Schema Normalization:** Transform EdNet / ASSISTments logs into AdaptiveLearn’s standardized feature vector format:
   $$\vec{x} = \left[ Q_{overall}, A_{topic}, T_{latency}, R_{opp\_count}, F_{mastery}, E_{remedial}, H_{prior\_recs} \right]$$
2. **Offline Pre-Training / Baseline Validation:** Train and cross-validate Candidate Models (Random Forest, LightGBM, XGBoost, MLP) on these normalized tables to determine the best-performing hyperparameters and feature weighting architectures.
3. **Synthetic Student Calibration:** Use the distributional parameters learned from ASSISTments/EdNet (e.g., typical forgetting rates, correlation between quiz latency and failure) to generate realistic synthetic student trajectories for local end-to-end testing before real user deployment.

---

# Part 2: Comprehensive Architecture Decision Records (ADRs)

Below are the 11 architectural decisions governing the system, exploring all viable possibilities, analyzing their trade-offs, and explaining why the selected approach is optimal.

---

### Decision 1: Topic Taxonomy & Concept Extraction
**The Problem:** The ML model must track performance at the topic level ($A, R$), but arbitrary uploaded PDFs have no predefined topic taxonomy.

#### Possibilities Evaluated:
* **Option 1A: Single Document-Level Topic (Current Repository State)**
  * *How it works:* The entire PDF filename is recorded as the topic (e.g., `topic = "Operating_Systems.pdf"`).
  * *Pros:* Zero implementation effort; zero LLM token cost.
  * *Cons:* Fatal pedagogical flaw. A student cannot "master" an entire 200-page book at once. Fails to locate whether the student understands *Page Tables* vs. *Deadlocks*.
* **Option 1B: Per-Chunk LLM Tagging Pass**
  * *How it works:* Every individual chunk is passed through an LLM prompt during ingestion: *"What topic does this chunk belong to?"*
  * *Pros:* Highly granular.
  * *Cons:* Prohibitive latency and cost. Ingesting a 50-page document (150 chunks) triggers 150 LLM API calls, instantly exhausting free-tier rate limits (Groq RPM/TPM caps). Results in a fractured, inconsistent topic list (e.g., "Deadlock", "Deadlocks", "Deadlock condition").
* **Option 1C: Document Outline Generation + Local Embedding Similarity (Chosen)**
  * *How it works:* 
    1. During PDF ingestion, run a single LLM call on the document table of contents or high-level summary to extract 5–15 canonical topics with short descriptions: `[{"topic_id": "T1", "name": "Deadlock Handling", "desc": "Banker's algorithm, prevention..."}]`.
    2. Embed each topic description using the local `sentence-transformers/all-MiniLM-L6-v2` model.
    3. Tag all document chunks by computing cosine similarity against the topic vectors (selecting the top 1–2 topics per chunk).
  * *Pros:* Requires only **1 LLM call** per document. Fast, deterministic, zero ongoing API cost. Guarantees a clean, standardized 5–15 topic taxonomy for the document.
* **Option 1D: Universal Multi-Document Cross-Ontology**
  * *How it works:* Merge topics across different PDFs (e.g., linking "Deadlocks" in Book A to "Deadlocks" in Book B).
  * *Pros:* Global cross-document learning.
  * *Cons:* Significant complexity; requires entity resolution and graph reconciliation. Overkill for v1.

**Decision & Rationale:** **Option 1C is the best path.** It solves the granularity problem with $O(1)$ LLM cost, operates within Groq rate limits, and provides a stable set of IDs (`topic_id`) that downstream quiz questions and flashcards inherit.

---

### Decision 2: Target Variable Formulation ($Y$) & ML Task Type
**The Problem:** Supervised machine learning requires a well-defined prediction target $Y$ mapped from input vector $X$.

#### Possibilities Evaluated:
* **Option 2A: 3-Class Discrete Classification (`high` / `medium` / `low` priority)**
  * *How it works:* The model predicts a categorical label for each topic.
  * *Pros:* Intuitive to describe conceptually.
  * *Cons:* Imprecise boundary definitions. In small sample sizes, models frequently collapse to predicting the majority class ("medium"). Does not provide an inherent intra-class ranking (if 5 topics are "high", which one should the student study first?).
* **Option 2B: Binary Classification (`needs_review` vs. `mastered`)**
  * *How it works:* Standard $0$ or $1$ prediction.
  * *Pros:* Clean mathematical boundary; high availability of metrics (ROC-AUC, Precision, Recall).
  * *Cons:* Binary labels lack nuance; fails to convey whether a topic is slightly shaky or completely forgotten.
* **Option 2C: Continuous Knowledge Deficit Score ($D \in [0.0, 1.0]$) via Proxy-Labeling (Chosen)**
  * *How it works:* Define the target as the student's predicted **Knowledge Deficit Score** (or probability of failure on the next encounter):
    $$Y = 1.0 - \text{Accuracy}_{\text{next\_attempt}} + \text{DecayPenalty}$$
    The model outputs a continuous probability $D \in [0.0, 1.0]$. Topics are ranked in descending order of $D$.
  * *Pros:* Continuous outputs allow dynamic thresholding and deterministic sorting. Top-$K$ recommendations can always be surfaced. Seamlessly unifies regression, classification, and priority ranking.
* **Option 2D: Learning-to-Rank (LTR / Pairwise LambdaMART)**
  * *How it works:* Predicts relative order between pairs of topics directly.
  * *Pros:* Highly tailored to recommendation lists.
  * *Cons:* Requires complex pairwise query group formatting; difficult to train with sparse early data.

**Decision & Rationale:** **Option 2C is the best path.** Continuous deficit scoring eliminates tie-breaking issues, allows the UI to show a clean percentage ("82% Deficit Risk"), and translates seamlessly into priority tiers (`high` if $D \ge 0.7$, `medium` if $0.4 \le D < 0.7$, `low` if $D < 0.4$).

---

### Decision 3: Cold-Start Handling Strategy
**The Problem:** On Day 1, a student has taken zero or only one quiz. Machine learning models cannot produce reliable inferences without interaction history.

#### Possibilities Evaluated:
* **Option 3A: Block Recommendations Until Minimum Quizzes Taken**
  * *How it works:* The app shows a lock screen: *"Complete 5 quizzes to unlock personalized recommendations."*
  * *Pros:* Ensures the ML model never runs on sparse data.
  * *Cons:* Severe user experience penalty; discourages new users and ruins the onboarding flow.
* **Option 3B: Unconditional Global Average Prior**
  * *How it works:* Recommend topics based solely on what other students found difficult.
  * *Pros:* Requires no data from the current user.
  * *Cons:* Completely unpersonalized. If student A aced Chapter 1, but the global cohort struggled with Chapter 1, student A is wrongly told to restudy Chapter 1.
* **Option 3C: Phased Hybrid: Deterministic Heuristic $\rightarrow$ Supervised ML Handoff (Chosen)**
  * *How it works:* 
    * **Phase A ($N < 3$ attempts):** Use a deterministic heuristic scoring formula:
      $$\text{Deficit} = 0.5 \cdot (1 - \text{Accuracy}) + 0.3 \cdot \text{RecencyPenalty} + 0.2 \cdot (1 - \text{FlashcardMastery})$$
    * **Phase B ($N \ge 3$ attempts across $\ge 2$ sessions):** The trained ML model (LightGBM/XGBoost) takes over scoring and ranking.
  * *Pros:* Zero user friction on Day 1; recommendations are personalized from the very first quiz; smoothly transitions to advanced ML as data volume matures.
* **Option 3D: Zero-Shot LLM Reasoning on Raw Logs**
  * *How it works:* Dump all raw logs into an LLM context window to pick the weak topic.
  * *Pros:* No training needed.
  * *Cons:* Token-expensive, slow, non-deterministic, and violates the requirement of having a real ML learner layer.

**Decision & Rationale:** **Option 3C is the best path.** It guarantees instant utility on Day 1 while providing a mathematically sound handoff threshold for machine learning.

---

### Decision 4: Model Granularity: Global Cohort vs. Per-Student vs. Per-Document
**The Problem:** Should we train one model per student, one per document, or one unified global model?

#### Possibilities Evaluated:
* **Option 4A: Per-Student Models**
  * *How it works:* Train an individual model for each registered user.
  * *Pros:* Hyper-fitted to an individual's personal quirks.
  * *Cons:* Catastrophic sample starvation. An individual student might only complete 10–20 quizzes in a month—nowhere near enough data to train an XGBoost or MLP model without severe overfitting. High maintenance overhead.
* **Option 4B: Per-Document Models**
  * *How it works:* Train a model specific to each uploaded book.
  * *Pros:* Captures content-specific difficulty.
  * *Cons:* Zero knowledge transfer across different subjects; cannot model student learning behavior.
* **Option 4C: Unified Global Model with Personalized Runtime Feature Vectors (Chosen)**
  * *How it works:* A single global model is trained across all historical $(student, topic, attempt)$ records. The model learns general learning dynamics (e.g., how accuracy drop and time decay correlate with failure). At inference time, personalization is achieved entirely through the student's unique feature vector $\vec{x}$.
  * *Pros:* Aggregates statistical power across all users; trains robust decision trees; inference requires a single fast call; standard industry architecture in recommender systems.

**Decision & Rationale:** **Option 4C is the best path.** It delivers true personalization through input features while maintaining the data volume required for stable machine learning.

---

### Decision 5: Data Storage Architecture: Relational vs. Vector vs. File
**The Problem:** The existing codebase stores quiz records in a flat CSV file (`data/audio/performance.csv`) and document embeddings in ChromaDB.

#### Possibilities Evaluated:
* **Option 5A: Flat CSV Files (Current State)**
  * *Pros:* Zero setup.
  * *Cons:* No relational integrity, no concurrent write safety, no indexing, easily corrupted, difficult to query multi-table relationships ($Q, A, T, R, F, E, H$).
* **Option 5B: Storing Learner State Inside ChromaDB Metadata**
  * *Pros:* Keeps everything in one database.
  * *Cons:* Vector databases are built for nearest-neighbor semantic search, not ACID-compliant relational joins. Tracking longitudinal student attempt histories inside vector metadata causes massive index bloat and query slowdowns.
* **Option 5C: SQLite (Local/MVP) with PostgreSQL Migration Path (Chosen)**
  * *Pros:* Serverless, zero configuration, built into Python (`sqlite3`), supports full SQL joins, ACID transactions, and easily handles millions of rows. Migrating to PostgreSQL in production requires changing only the connection string (via SQLAlchemy).
* **Option 5D: Cloud NoSQL (MongoDB / DynamoDB)**
  * *Pros:* Flexible JSON schema.
  * *Cons:* Relational queries across attempts, questions, and topics require complex client-side aggregation or redundant denormalization.

**Decision & Rationale:** **Option 5C is the best path.** Clean separation: ChromaDB stores document vectors; SQLite stores structured learner interactions and relational foreign keys.

---

### Decision 6: Machine Learning Algorithm Selection
**The Problem:** Which model family should drive the recommendation engine? (Candidates: Random Forest, XGBoost, LightGBM, MLP, Deep Knowledge Tracing).

#### Possibilities Evaluated:
* **Option 6A: Deep Knowledge Tracing (DKT / LSTM / Transformers)**
  * *Pros:* Native sequential modeling of student knowledge state.
  * *Cons:* Requires massive training datasets ($10^5+$ sequences) to converge without overfitting; heavy GPU dependency; complex deployment footprint.
* **Option 6B: Multi-Layer Perceptron (MLP)**
  * *Pros:* Universal function approximator; capable of non-linear representations.
  * *Cons:* Sensitive to feature scaling; prone to overfitting on small tabular datasets; uninterpretable "black box".
* **Option 6C: LightGBM / XGBoost with Random Forest Baseline (Chosen)**
  * *Pros:* State-of-the-art performance on tabular behavioral data; handles missing values natively; invariant to feature scaling; lightning-fast training on CPU; native integration with **SHAP** for exact feature attribution.
* **Option 6D: Logistic / Ridge Regression**
  * *Pros:* Highly interpretable.
  * *Cons:* Cannot capture non-linear interactions (e.g., high quiz score but long time decay).

**Decision & Rationale:** **Option 6C is the best path.** Gradient-boosted decision trees dominate tabular feature learning, run efficiently in lightweight environments, and provide gold-standard explainability via TreeSHAP.

---

### Decision 7: Spaced Repetition / Memory Retention Engine
**The Problem:** How to mathematically quantify memory decay ($R$) and flashcard mastery ($F$) over time.

#### Possibilities Evaluated:
* **Option 7A: Naive Time Counter (Days Since Last Seen)**
  * *Pros:* Trivial to calculate.
  * *Cons:* Ignores difficulty and student recall history. Reviewing a card once 5 days ago is very different from reviewing it 10 times consecutively.
* **Option 7B: SuperMemo SM-2 Algorithm (Chosen)**
  * *How it works:* Industry-standard algorithm used in Anki:
    * Updates an Easiness Factor ($EF$) based on student recall quality ($0$ to $5$).
    * Dynamically scales review intervals: $I(1) = 1\text{ day}, I(2) = 6\text{ days}, I(n) = I(n-1) \cdot EF$.
  * *Pros:* Computationally trivial, backed by decades of cognitive research, provides normalized mastery and retention metrics directly into feature $F$.
* **Option 7C: FSRS (Free Spaced Repetition Scheduler)**
  * *Pros:* More accurate modern alternative based on DSR (Difficulty, Stability, Retrievability) model.
  * *Cons:* Slightly more complex parameter optimization; SM-2 is more than sufficient for v1.

**Decision & Rationale:** **Option 7B (SM-2) is the best path.** It is proven, lightweight, and supplies clean mathematical signals for the feature vector.

---

### Decision 8: Evaluation Protocol & Data Split Strategy
**The Problem:** Behavioral student data is sequential and time-ordered. An improper validation split leads to data leakage and false metrics.

#### Possibilities Evaluated:
* **Option 8A: Standard K-Fold Cross-Validation (Random Shuffling)**
  * *Pros:* Easy one-line scikit-learn call.
  * *Cons:* **Catastrophic data leakage.** Random shuffling puts future quiz attempts in the training set and past attempts in the test set. Models learn to "cheat" using future knowledge, yielding inflated, meaningless accuracy scores.
* **Option 8B: GroupKFold by Student ID**
  * *Pros:* Tests generalization to completely unseen students.
  * *Cons:* Does not evaluate how well the model tracks an existing student's learning trajectory over time.
* **Option 8C: Temporal (Time-Based) Split combined with Out-of-Time Group Holdout (Chosen)**
  * *How it works:* Split data strictly by chronological timestamp. Train on interactions from $T_0$ to $T_{\text{split}}$, validate on interactions from $T_{\text{split}}$ to $T_{\text{now}}$.
  * *Pros:* Faithfully mimics production inference: the model is always asked to predict *future* performance based on *past* behavior.

**Decision & Rationale:** **Option 8C is the best path.** It prevents temporal leakage and produces reliable performance metrics.

---

### Decision 9: Explainability & Model Attribution (XAI)
**The Problem:** Students do not trust or act upon unexplained recommendations ("Study Page Tables"). They need to understand *why*.

#### Possibilities Evaluated:
* **Option 9A: Hardcoded Rule Templates**
  * *Pros:* Simple.
  * *Cons:* Static and repetitive; disconnected from the actual ML model's weights.
* **Option 9B: Freeform LLM Explanation Generation**
  * *Pros:* Eloquent text.
  * *Cons:* Hallucination-prone. The LLM might claim "you studied this 2 days ago" when the database shows it was 10 days ago.
* **Option 9C: TreeSHAP Attribution $\rightarrow$ Dynamic Natural Language Template (Chosen)**
  * *How it works:* 
    1. Compute exact SHAP values for the top 3 contributing features driving the recommendation (e.g., $SHAP(A_{\text{topic}}) = +0.42$, $SHAP(R_{\text{recency}}) = +0.31$).
    2. Inject these verified quantitative facts into a dynamic UI explanation: *"Deadlocks is marked High Priority because your quiz accuracy was 33% and you haven't reviewed it in 6 days."*
  * *Pros:* 100% grounded in model reality; eliminates hallucination; builds student trust.

**Decision & Rationale:** **Option 9C is the best path.** It combines mathematical rigor (TreeSHAP) with clear, actionable human communication.

---

### Decision 10: Service Decoupling & Software Architecture
**The Problem:** The current app mixes Streamlit UI callbacks, agent reasoning, and data persistence in monolithic files (`ui/app.py`).

#### Possibilities Evaluated:
* **Option 10A: Keep Logic Embedded in Streamlit Callbacks**
  * *Pros:* No refactoring needed right now.
  * *Cons:* Impossible to write automated unit tests for the recommender; cannot swap Streamlit for FastAPI or Next.js later without a total rewrite.
* **Option 10B: Microservices with REST/gRPC**
  * *Pros:* Fully isolated services.
  * *Cons:* Extreme over-engineering for this stage; complex multi-process orchestration.
* **Option 10C: Decoupled Python Service Layer (Chosen)**
  * *How it works:* Organize the codebase into modular, framework-agnostic Python packages:
    * `database/`: SQLAlchemy/SQLite models and queries.
    * `engine/recommender.py`: Feature computation, rule scorer, and ML model inference.
    * `engine/spaced_repetition.py`: Pure SM-2 math.
    * `ui/`: Pure Streamlit presentation consuming the service layer.
  * *Pros:* Enables unit testing; allows the ML pipeline to run as a standalone batch job; allows migrating to FastAPI or Next.js later with zero backend changes.

**Decision & Rationale:** **Option 10C is the best path.** It achieves clean architectural modularity without unnecessary distributed system overhead.

---

### Decision 11: Cross-Document Topic Merging Strategy
**The Problem:** What happens if a student uploads two different PDFs that cover the same underlying subject?

#### Possibilities Evaluated:
* **Option 11A: Scoped to Single Document for v1 (Chosen)**
  * *How it works:* Topics exist strictly within the scope of an uploaded document (`topic_id` belongs to `document_id`).
  * *Pros:* Fast to build, eliminates complex entity resolution, guarantees reliable scoping.
* **Option 11B: Cross-Document Topic Merging via Graph Embeddings (Target for v2)**
  * *How it works:* Cluster topics across documents using embedding distance thresholds and maintain a global knowledge graph.
  * *Pros:* Long-term holistic student profile.
  * *Cons:* Introduces edge cases where similar names mean different things in different contexts.

**Decision & Rationale:** **Option 11A for v1, transitioning to 11B in v2.** Shipping v1 per-document enables immediate deployment while establishing the exact schema needed for cross-document linking later.

---

# Part 3: Architecture Decision Summary Matrix

| Decision Area | Chosen Path | Primary Alternative Considered | Decisive Advantage of Chosen Path |
| :--- | :--- | :--- | :--- |
| **1. Topic Extraction** | **Document Outline LLM + Cosine Vector Tagging** | Per-chunk LLM Tagging | 1 LLM call vs 150+ calls; eliminates rate limits and maintains clean taxonomy. |
| **2. Target $Y$ Formulation** | **Continuous Deficit Score ($D \in [0, 1]$)** | 3-Class (`high/med/low`) Categorization | Eliminates class collapse; enables continuous ranking and clean top-$K$ recommendations. |
| **3. Cold-Start Handling** | **Heuristic Scorer $\rightarrow$ ML Handoff ($N \ge 3$)** | Blocking / Gating Access | High-quality Day 1 user onboarding with seamless transition to ML. |
| **4. Model Scope** | **Unified Global Model + Personalized Features** | Per-Student Models | Solves sample starvation; statistical power across users with individual personalization. |
| **5. Training Data Strategy** | **Public Benchmarks $\rightarrow$ Synthetic $\rightarrow$ Organic** | Organic Data Only | Unblocks model development immediately without waiting months for real users. |
| **6. Storage Engine** | **Relational SQLite (Path to PostgreSQL)** | ChromaDB Metadata / CSV | ACID integrity, foreign-key relationships, and fast multi-table joins. |
| **7. ML Model Family** | **LightGBM / XGBoost + Random Forest Baseline** | Deep Knowledge Tracing (DKT/LSTM) | Dominant performance on tabular features, CPU efficiency, native TreeSHAP integration. |
| **8. Memory Retention** | **SuperMemo SM-2 Algorithm** | Naive Days Elapsed Counter | Grounded in cognitive science; dynamic interval and ease scaling. |
| **9. Evaluation Protocol** | **Chronological Temporal Split** | Random K-Fold CV | Eliminates temporal data leakage and reflects true real-world predictive power. |
| **10. Explainability** | **TreeSHAP Feature Attribution $\rightarrow$ Dynamic Copy** | Unconstrained LLM Text | Mathematically grounded explanations without LLM hallucinations. |
| **11. Code Architecture** | **Decoupled Python Service Layer** | UI Callback Monolith | Testable, modular, and ready for future FastAPI / Next.js migration. |

---

# Part 4: Implementation Phasing Roadmap

```
           PHASE 1 (Week 1)                     PHASE 2 (Week 2)
      +------------------------+           +------------------------+
      |    DATA FOUNDATION     |           | RULE-BASED RECOMMENDER |
      | - SQLite Schema        | --------> | - SM-2 Retention Logic |
      | - Topic Extraction     |           | - Cold-Start Formula   |
      | - Event Logging        |           | - Remedial UI Hook     |
      +------------------------+           +------------------------+
                  |                                     |
                  v                                     v
           PHASE 3 (Week 3-4)                   PHASE 4 (Week 5)
      +------------------------+           +------------------------+
      |        ML LAYER        |           |  DASHBOARD & RETRAIN   |
      | - Train on Benchmarks  | --------> | - Automated Retraining |
      | - XGBoost / LightGBM   |           | - SHAP Explainability  |
      | - Temporal Validation  |           | - Mastery Analytics    |
      +------------------------+           +------------------------+
```
