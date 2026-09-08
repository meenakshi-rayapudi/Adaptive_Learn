# 🎓 Veras: Autonomous AI Academic Tutor & Educational Agent

[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq%20Llama%203.3-orange.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)

**Veras** is a 100% autonomous, agentic AI educational companion that transforms unstructured learning materials (PDF textbooks, course notes, and YouTube video lectures) into structured, interactive, and multi-modal study assets.

---

## 🏛️ Autonomous Agent Architecture

Unlike conventional scripted educational tools, Veras operates via an **autonomous ReAct tool-calling agent brain**. The agent autonomously evaluates user requests, determines the required cognitive tools, performs semantic searches over the document vector index, and dynamically synthesizes study materials.

`
+-------------------------------------------------------------+
|               Student Chat Prompt / UI Directives           |
+-------------------------------------------------------------+
                              |
                              v
                +----------------------------+
                |  🧠 Autonomous TutorAgent  |
                +----------------------------+
                              |
      +-----------------------+-----------------------+
      |                       |                       |
      v                       v                       v
+------------------+  +--------------------+  +------------------+
| document_search  |  | flashcard_creator  |  |   quiz_creator   |
+------------------+  +--------------------+  +------------------+
      |                       |                       |
      v                       v                       v
+------------------+  +--------------------+  +------------------+
|  study_planner   |  | adaptive_remedial  |  |  audio_narrator  |
+------------------+  +--------------------+  +------------------+
                              |
                              v
              +--------------------------------+
              | 📦 Agent Artifact Store & Sync |
              +--------------------------------+
`

---

## ✨ Key Features & Autonomous Capabilities

### 1. 🔍 Autonomous Knowledge Ingestion & RAG Retrieval
- **Multi-Source Ingestion:** Ingests local PDFs, TXT, Markdown notes via Microsoft MarkItDown and remote YouTube video lectures via youtube-transcript-api.
- **Zero-Cost Local RAG:** Dense vector embeddings using sentence-transformers/all-MiniLM-L6-v2 stored in local ChromaDB.
- **Tool:** document_search(query: str) retrieves top-k contextual excerpts autonomously.

### 2. 🗂️ Active Recall Flashcards & Spaced Repetition (SR-Lite)
- **3D Card Flip UI:** Interactive card flipping built with pure CSS.
- **Spaced Repetition Algorithm:** Time-decay priority queue (New -> Need Review (1m interval) -> Mastered (60m interval)).
- **Neural Audio Pronunciation:** Listen to any flashcard with one click.
- **Tool:** lashcard_creator(topic: str, num_cards: int) generates structured JSON decks.

### 3. 📝 Interactive Assessments & Cognitive Remediation
- **Multi-Format Testing:** Generates Multiple Choice (MCQ) and True/False questions with ground-truth explanations.
- **Autonomous Error Diagnosis:** When a student misses questions, the agent diagnoses root conceptual misconceptions and generates a custom Remedial Study Guide with memory mnemonics.
- **Tools:** quiz_creator and daptive_remedial_evaluator.

### 4. 🗓️ Proactive 3-Day Mastery Planning
- **Pedagogical Strategy:** Analyzes document complexity and breaks learning down into:
  - **Day 1:** Core Foundations & Key Terminology
  - **Day 2:** Deep Dive & Complex Mechanisms
  - **Day 3:** Synthesis, Practical Application & Assessment Check
- **Tool:** study_planner(topic: str).

### 5. 🎧 Neural Text-to-Speech Audio Lessons
- **Neural TTS:** Powered by Microsoft edge-tts (en-US-AriaNeural).
- **In-App Streaming & Download:** Converts agent summaries or full study guides into downloadable .mp3 audio lessons.
- **Tool:** udio_narrator(text: str).

### 6. 📊 Learning Analytics & Mastery Dashboard
- Persistent performance logging in data/audio/performance.csv.
- Visual progress charts: accuracy trends over time, daily study frequency, topic mastery distribution, and cumulative score metrics.

---

## 📁 Repository Structure

`
Educational_Content_Agentic/
├── core/
│   ├── config.py             # Environment & model configurations
│   ├── engine.py             # Autonomous TutorAgent orchestrator & ReAct loop
│   ├── provider.py           # ChatGroq LLM provider factory
│   └── vector_store.py       # ChromaDB vector store & embeddings
├── models/
│   ├── schemas.py            # Pydantic schemas (Flashcards, Quizzes, Plans, Remediation)
│   └── quiz.py               # Schema exports & backwards compatibility
├── tools/
│   ├── retrieval_tool.py     # Semantic document retrieval tool
│   ├── flashcard_tool.py     # Structured flashcard generator
│   ├── quiz_tool.py          # Structured assessment generator
│   ├── planner_tool.py       # 3-day mastery plan generator
│   ├── adaptive_tool.py      # Cognitive diagnosis & remedial guide generator
│   ├── audio_tool.py         # Neural TTS audio narration tool
│   ├── parser.py             # MarkItDown document conversion & safe JSON parsers
│   ├── chunker.py            # RecursiveCharacterTextSplitter chunking
│   └── youtube_tool.py       # YouTube transcript extraction
├── ui/
│   ├── app.py                # Main Streamlit web application & goal dispatcher
│   ├── flashcard_ui.py       # Spaced repetition 3D card grid
│   ├── components/           # Modular UI headers, sidebars, and chat interface
│   └── styles/css.py         # Custom CSS animations & design
├── utils/
│   ├── analytics.py          # CSV performance logging & Plotly/Streamlit charts
│   ├── audio_utils.py        # edge-tts async audio pipeline
│   └── youtube_utils.py      # Video transcript utility helpers
├── tests/
│   └── test_agent_system.py  # Automated unit tests for schemas, parser, and tools
├── requirements.txt          # Python project dependencies
└── main.py                   # Autonomous CLI tutor interface
`

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Configure Environment Variables (.env)
`env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.2
`

### 3. Run the Streamlit Application
`ash
streamlit run ui/app.py
`

### 4. Run the Autonomous CLI Interface
`ash
python main.py
`
