# AdaptiveLearn — Master Study Guide & 6-Week 3-Person Development Roadmap

## Part 1: Topics to Master to Build and Defend This Project in an Interview

To build this project confidently and explain it to senior software engineers, ML interviewers, or professors, you need a strong conceptual grasp across **six core domains**. Below is each topic, the exact mental model, and how to articulate it in an interview.

---

### Domain 1: System Architecture — The "Two-Brain" Paradigm
* **The Core Concept:** Most educational apps are simple "RAG wrappers" (they understand documents, but treat every student interaction as stateless). AdaptiveLearn uses a **Dual-Engine Architecture**:
  * **Engine 1 (Content Brain - RAG & LLM):** Understands the unstructured textbook (extracts text, generates questions, explains concepts).
  * **Engine 2 (Learner Brain - Relational DB & ML):** Understands the student (tracks historical performance, measures forgetting curves, predicts weak topics).
  * **Recommendation Engine:** Bridges both brains—ML picks *what concept* needs attention; RAG fetches the *exact content* to teach it.
* **How to Explain in an Interview:**
  > *"Most AI tutors today have amnesia. They can explain quantum mechanics from a PDF, but they don't remember that yesterday you scored 20% on Wave Equations and haven't reviewed it in 5 days. We decoupled content understanding (handled via vector embeddings and LLM tool calling) from learner state modeling (handled via a relational database, an SM-2 retention engine, and a LightGBM/XGBoost predictor). This allows the system to guide the student proactively rather than waiting for passive questions."*

---

### Domain 2: RAG & Document Intelligence
* **Document Parsing & Normalization:** Converting noisy PDFs/videos into clean markdown using AST parsers (`markitdown`, `youtube-transcript-api`).
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` with chunk size ~1000 characters and 150-character overlap to preserve semantic context across chunk boundaries.
* **Dense Vector Embeddings & Vector DB:**
  * Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
  * Cosine Similarity: Measuring directional angle between query vector and chunk vectors in inner-product space.
  * Local Vector Indexing: ChromaDB storing chunk vectors and metadata.
* **Topic Taxonomy Extraction (The Key Innovation):**
  * Rather than burning 150 LLM calls per document, generate 5–15 canonical topic tuples `(topic_id, name, desc)` once from the document summary/outline, embed them, and assign chunks via local cosine similarity.

---

### Domain 3: Machine Learning for Tabular Behavioral Data
* **Feature Engineering ($X = \{Q, A, T, R, F, E, H\}$):**
  * $Q$: Global recent quiz performance (normalized ratio).
  * $A$: Topic-specific historical accuracy.
  * $T$: Cognitive struggle / latency in milliseconds (log-transformed).
  * $R$: Days elapsed since last study event (spaced decay).
  * $F$: Flashcard mastery score from SM-2 intervals.
  * $E$: Engagement intensity (remedial guides read, questions asked).
  * $H$: Historical recommendation adherence.
* **Why Tree-Based Models (XGBoost, LightGBM, Random Forest) over Deep Learning:**
  * Scale invariance across millisecond latencies and accuracy percentages.
  * Native routing of missing data ($\text{NaN}$) for brand-new modalities.
  * Research backing: *Grinsztajn et al. (NeurIPS 2022)* proving GBDTs outperform neural nets on tabular data.
* **Proxy-Labeling for the Target ($Y$):**
  * Supervised learning requires labels. Without human annotators, we derive $Y$ as a continuous **Knowledge Deficit Score** $D \in [0.0, 1.0]$ based on next-attempt failure or performance decay.

---

### Domain 4: Cognitive Science & Spaced Repetition (SM-2)
* **The Ebbinghaus Forgetting Curve:** Memory retention decays exponentially over time: $R(t) = e^{-\frac{t}{S}}$, where $S$ is memory stability.
* **SuperMemo SM-2 Algorithm:**
  * Tracks an **Easiness Factor ($EF$)** initialized at $2.5$:
    $$EF' = EF + (0.1 - (5 - q) \cdot (0.08 + (5 - q) \cdot 0.02))$$
  * Dynamically schedules review intervals: $I(1) = 1\text{ day}, I(2) = 6\text{ days}, I(n) = I(n-1) \cdot EF$.
  * Converts discrete flashcard reviews into continuous mastery signals ($F$) and overdue risk ($R$).

---

### Domain 5: Explainable AI (XAI) & TreeSHAP
* **The Problem:** Black-box recommendations demoralize students. Telling someone "Study Topic X" without justification causes distrust.
* **Shapley Additive Explanations (SHAP):**
  * Fairly distributes the prediction payout among all input features based on cooperative game theory.
  * **TreeSHAP:** Exact, polynomial-time $O(TLD^2)$ algorithm for tree ensembles.
* **Translating Math into Pedagogy:**
  * Top positive SHAP features $\rightarrow$ Dynamic student-facing text:
    *"Deadlocks was prioritized because your recent quiz accuracy is 33% and it has been 6 days since your last flashcard drill."*

---

### Domain 6: ML Evaluation Protocols & Data Integrity
* **Temporal / Chronological Validation:**
  * Never use standard random K-Fold on sequential student data. Shuffling future attempts into the training set causes **temporal data leakage**, creating unrealistically high accuracy that collapses in production.
  * Always split data chronologically: train on $T_0 \to T_{\text{split}}$, test on $T_{\text{split}} \to T_{\text{now}}$.
* **Metrics:**
  * **ROC-AUC & PR-AUC:** Measures discrimination between struggling vs. mastered concepts.
  * **Brier Score:** Measures how well-calibrated the predicted probability of failure is.

---

# Part 2: 6-Week 3-Person Collaborative Development Roadmap

### Team Role Allocation

```
+-------------------------------------------------------------------------------------------------+
|                                       TEAM RESPONSIBILITY MATRIX                                |
+-------------------------------------------------------------------------------------------------+
| Role              | Title                                | Core Tech Stack                      |
+-------------------+--------------------------------------+--------------------------------------+
| Person 1 (ML)     | ML & Knowledge Modeling Engineer     | Python, Scikit-Learn, LightGBM,      |
|                   |                                      | XGBoost, SHAP, Pandas, SM-2 Math     |
+-------------------+--------------------------------------+--------------------------------------+
| Person 2 (Backend)| Backend, Database & RAG Engineer     | Python, SQLite, SQLAlchemy,          |
|                   |                                      | LangChain, Groq API, ChromaDB, AST   |
+-------------------+--------------------------------------+--------------------------------------+
| Person 3 (UI/UX)  | Frontend, UI & Analytics Engineer    | Streamlit, Plotly, HTML/CSS,         |
|                   |                                      | Session State, Audio Streaming       |
+-------------------------------------------------------------------------------------------------+
```

---

### Week 1: Data Architecture, Topic Extraction & Relational Foundation

#### 🎯 Week 1 End Goal:
A student uploads a PDF $\rightarrow$ The system automatically extracts 5–15 canonical topics, tags all chunks, stores them in ChromaDB, initializes the SQLite database schema, and renders the extracted topics cleanly on the UI sidebar.

```
DEVELOPMENT FLOW - WEEK 1:
Person 2 (Designs SQLite Schema & Topic Extraction Tool)
   │
   ├──> Pushes `database/schema.py` & `core/topic_extractor.py`
   │
   ├──> Person 1 pulls schema: Writes synthetic student generator script (`seed_data.py`)
   │
   └──> Person 3 pulls extractor: Updates sidebar UI to display detected document topics
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Study the ASSISTments and EdNet schemas.
  * Write `scripts/seed_student_data.py`: A generator that produces 50 realistic synthetic student profiles (e.g., *Fast Learner*, *Struggling Student*, *Crammer*) across 30 days.
  * Define the feature vector calculation specification in math/code.
* **Person 2 (Backend / RAG):**
  * Build the relational schema in SQLite using SQLAlchemy: `students`, `documents`, `topics`, `chunks`, `quiz_attempts`, `quiz_questions`, `flashcard_events`, `recommendations`.
  * Implement `core/topic_extractor.py`: 1 LLM call to extract document topics + cosine similarity matching to tag all chunks during PDF ingestion.
  * Update `core/vector_store.py` to persist `topic_id` in ChromaDB chunk metadata.
* **Person 3 (Frontend / UI):**
  * Clean up `ui/app.py`: Decouple monolith code into reusable components.
  * Update document upload sidebar: When a PDF is ingested, render the extracted topics with color badges and chunk counts.
  * Implement unified session-state management for `student_id` and active `document_id`.

---

### Week 2: Topic-Tagged Generation & Cold-Start Recommender

#### 🎯 Week 2 End Goal:
Quiz questions and flashcards are generated with attached `topic_id`s. Quiz submissions log directly to SQLite. A brand-new student immediately receives their first data-driven remedial recommendation via the Day 1 Heuristic Formula.

```
DEVELOPMENT FLOW - WEEK 2:
Person 2 (Updates Quiz & Flashcard Tools with Pydantic topic_id)
   │
   ├──> Person 3 integrates submission forms to log every answer to SQLite
   │
   └──> Person 1 implements the Cold-Start Weighted Scorer; Person 3 renders the Rec Banner
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Implement `engine/cold_start.py`: The deterministic Day 1 heuristic formula:
    $$\text{Deficit} = 0.5 \cdot (1 - \text{Accuracy}) + 0.3 \cdot \text{RecencyPenalty} + 0.2 \cdot (1 - \text{FlashcardMastery})$$
  * Implement `engine/spaced_repetition.py`: The core SuperMemo SM-2 algorithm functions (`update_card_review`, calculating ease factor $EF$ and next interval $I$).
* **Person 2 (Backend / RAG):**
  * Update `models/schemas.py`: Ensure `QuizQuestionItem` and `FlashcardItem` carry mandatory `topic_id`.
  * Update `tools/quiz_tool.py` and `tools/flashcard_tool.py` so questions are generated per topic or tagged with the nearest `topic_id`.
  * Build database logging methods: `db.log_quiz_attempt()`, `db.log_question_event()`, `db.log_flashcard_review()`.
* **Person 3 (Frontend / UI):**
  * Connect quiz submission buttons to the SQLite logging service (replacing the old flat CSV write).
  * Redesign the Flashcard UI to display SM-2 quality ratings (buttons: *Hard*, *Good*, *Easy* instead of just flip).
  * Build a "Recommended Next Topic" banner on the top of the chat and quiz tabs that renders the cold-start output.

---

### Week 3: Multi-Session Persistence & Benchmark Data Pre-Training

#### 🎯 Week 3 End Goal:
The platform accurately tracks student performance across multiple browser sessions. Person 1 has trained baseline models on normalized public benchmark data (ASSISTments/EdNet) and established evaluation baselines.

```
DEVELOPMENT FLOW - WEEK 3:
Person 1 (Trains baseline models on EdNet/ASSISTments and validates feature vector)
   │
   ├──> Delivers pre-trained model weights & scaler pipeline
   │
   ├──> Person 2 builds `core/learner_service.py` to aggregate raw DB logs into feature vectors
   │
   └──> Person 3 builds multi-session student selector (login/switch profile dropdown)
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Download and normalize an ASSISTments or EdNet sample into AdaptiveLearn's feature vector format $\vec{x} = [Q, A, T, R, F, E, H]$.
  * Implement `engine/features.py`: Aggregation functions that calculate $Q, A, T, R, F, E, H$ from raw database tables for any given `(student_id, topic_id)`.
  * Build the chronological temporal train/test split pipeline ($80\%$ early attempts, $20\%$ late attempts).
* **Person 2 (Backend / RAG):**
  * Implement `core/learner_service.py`: High-level API service that interfaces between the database, the agent, and the ML engine.
  * Build a targeted quiz generator tool: `quiz_creator_targeted(topic_id, difficulty)` so the tutor can generate a quiz specifically on a student's weak topic.
  * Add study session duration tracking (logging timestamps when the user reads summaries or chats).
* **Person 3 (Frontend / UI):**
  * Add a student profile switcher / input in the sidebar so multiple users or simulated test profiles can be tested.
  * Implement active study session timer in Streamlit state.
  * Build the UI for "Targeted Remedial Drill": Clicking "Review Weak Topic" triggers a focused 5-question drill on that exact topic.

---

### Week 4: The Model Tournament & SHAP Explainability

#### 🎯 Week 4 End Goal:
The ML model tournament is executed. The winning model (LightGBM/XGBoost) is packaged and integrated. When the student reaches $\ge 3$ attempts, the ML model replaces the cold-start heuristic, and each recommendation displays a clear SHAP-driven "Why" explanation.

```
DEVELOPMENT FLOW - WEEK 4:
Person 1 (Runs Model Tournament: Logistic Reg vs MLP vs RF vs LightGBM vs XGBoost)
   │
   ├──> Selects winner, serializes model (`model.pkl`), writes `engine/explainer.py` (TreeSHAP)
   │
   ├──> Person 2 integrates model inference into `engine/recommender.py` with cold-start fallback
   │
   └──> Person 3 builds the "Why this is recommended" UI card with dynamic SHAP bullet points
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Run the formal **Model Tournament** across Logistic Regression, MLP, Random Forest, LightGBM, and XGBoost using temporal splits.
  * Produce the model evaluation comparison table (ROC-AUC, PR-AUC, Brier score, Latency).
  * Integrate `shap.TreeExplainer` to compute exact local attributions for the top 3 contributing features.
  * Package the inference pipeline (`predict_topic_deficit(student_id, topic_id)`).
* **Person 2 (Backend / RAG):**
  * Implement the automated handoff logic:
    * If `student_attempts < 3`: call `cold_start_score()`.
    * If `student_attempts >= 3`: call `ml_predict_deficit()`.
  * Store recommendations in SQLite with `model_version` and SHAP explanation metadata.
  * Add automated error handling and fallback to prevent UI crashes if ML dependencies fail.
* **Person 3 (Frontend / UI):**
  * Design and build the **"Recommendation Breakdown" Card**:
    * Displays Topic Title, Priority Badge (High/Med/Low), and Deficit Risk gauge.
    * Displays dynamic, plain-English bullet points generated from SHAP values:
      *"Why: Quiz accuracy dropped to 30% on this topic AND it has been 7 days since last review."*
  * Add interactive action buttons: "Launch Flashcard Drill" and "Take Targeted Quiz".

---

### Week 5: Advanced Learning Analytics Dashboard & Closed-Loop Remediation

#### 🎯 Week 5 End Goal:
Tab 4 (Learning Analytics) is overhauled with interactive Plotly visualizations showing topic mastery, forgetting curves, and longitudinal progress. The proactive study loop is fully closed.

```
DEVELOPMENT FLOW - WEEK 5:
Person 2 (Exposes relational analytics queries in `core/learner_service.py`)
   │
   ├──> Person 1 provides model performance calibration curves & feature importance charts
   │
   └──> Person 3 builds the rich multi-chart Plotly Analytics Dashboard in Tab 4
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Build an automated model evaluation script that recalibrates feature importances and validates against new organic attempts.
  * Create a global model interpretability visualization (summary SHAP bee-swarm plot) to export for documentation/presentation.
* **Person 2 (Backend / RAG):**
  * Write optimized SQL aggregate queries for student analytics:
    * Accuracy trends by topic over time.
    * Flashcard retention breakdown (Count of New, Review, and Mastered cards).
    * Weekly study time distribution.
  * Connect the ReAct agent prompt to the learner model: the tutor agent can now inspect the student's weak topics during chat conversations.
* **Person 3 (Frontend / UI):**
  * Overhaul Tab 4 into an executive **Learning Analytics Dashboard**:
    * Visual 1: Topic Mastery Radar or Horizontal Bar Chart.
    * Visual 2: Spaced Repetition Retention Decay Curve.
    * Visual 3: Chronological Accuracy Trend Line (Plotly).
    * Visual 4: Study Time & Activity Heatmap.

---

### Week 6: Testing, Polish, Live Deployment & Interview Prep

#### 🎯 Week 6 End Goal:
The complete application is end-to-end stress-tested, deployed to Streamlit Cloud, documented with architecture diagrams, and recorded in a 5-minute showcase video.

```
DEVELOPMENT FLOW - WEEK 6:
All 3 Team Members
   ├──> Person 1: Finalizes model benchmarks, writes ML section of report & demo slides
   ├──> Person 2: Unit tests all tools, verifies Groq rate-limit handling, sets up deployment
   └──> Person 3: UI polishing, creates demo walkthrough script, records video presentation
```

#### Detailed Tasks:
* **Person 1 (ML / Data):**
  * Finalize the Model Tournament benchmark documentation in `docs/model_selection_and_benchmarks.md`.
  * Prepare the technical explanation slides on the ML architecture (Feature engineering, GBDTs, SM-2, SHAP).
* **Person 2 (Backend / RAG):**
  * Write end-to-end integration tests in `tests/test_learner_pipeline.py`.
  * Ensure smooth database migrations and zero-error cold starts on clean environments.
  * Configure environment secrets for cloud deployment.
* **Person 3 (Frontend / UI):**
  * Polish styling, responsiveness, and empty-state handling across all tabs.
  * Deploy the application live to **Streamlit Cloud**.
  * Draft the demo script and record the 5-minute video walkthrough showing the full student journey.

---

# Part 3: Cheat Sheet — Top 5 Interview Questions & Winning Answers

### Q1: "Why use Machine Learning here? Couldn't you just write a few IF-ELSE rules?"
* **Winning Answer:**
  > *"Rules work well for Day 1 cold start, which is exactly why we implemented a deterministic heuristic for new users. However, rules fail to scale across complex, non-linear feature interactions. For instance, is a student who scored 80% on a quiz 14 days ago in greater danger of forgetting than someone who scored 50% yesterday but has reviewed 10 flashcards? A human cannot tune static weights for these multi-variable trade-offs without bias. A gradient boosted tree learns the empirical failure boundary from historical patterns and provides exact SHAP values to explain the decision."*

### Q2: "Why didn't you use a Deep Learning model or an LSTM like Deep Knowledge Tracing (DKT)?"
* **Winning Answer:**
  > *"DKT is great when you have a static curriculum and 100,000+ interaction sequences from platforms like ASSISTments. But our platform allows users to upload arbitrary PDFs, meaning topics and questions are generated dynamically on the fly. Tabular tree models (XGBoost/LightGBM) generalize far better across dynamic concepts because they operate on domain-agnostic behavioral signals—accuracy, latency, forgetting decay, and engagement. Plus, as proven by Grinsztajn et al. at NeurIPS 2022, GBDTs consistently outperform deep learning on tabular data while requiring zero GPU overhead and providing exact TreeSHAP attribution."*

### Q3: "How did you prevent data leakage during model training?"
* **Winning Answer:**
  > *"Because student learning is chronological, standard random K-Fold cross-validation is disastrous—it allows future attempts to leak into the training set, giving the model hindsight knowledge and creating artificially inflated accuracy. We strictly used a chronological temporal split: training on early interaction sequences and validating strictly on future attempts. This simulates the exact conditions the model faces in production."*

### Q4: "What happens if a student uploads a 100-page book? Doesn't topic extraction blow up your LLM rate limits?"
* **Winning Answer:**
  > *"That was a critical design decision. If you prompt an LLM for every single chunk, you trigger hundreds of calls and exhaust API quotas immediately. We solved this with a hybrid approach: we call the LLM once on the document outline or high-level summary to extract 5–15 canonical topics with short descriptions. Then, we embed those topic descriptions and assign all chunks using fast, local cosine similarity with sentence-transformers. It gives us granular topic tagging with O(1) LLM cost."*

### Q5: "How does the system ensure the student actually follows recommendations?"
* **Winning Answer:**
  > *"We closed the recommendation loop through feature H (historical recommendation adherence) and targeted drills. When a topic is recommended, the UI doesn't just show text—it provides an interactive 'Launch Targeted Drill' button. This prompts the RAG layer to generate a 5-question assessment strictly drawn from that topic's chunk IDs. Once the student completes it, the database logs the new attempt, recalculates the deficit score, and updates the dashboard in real time."*
