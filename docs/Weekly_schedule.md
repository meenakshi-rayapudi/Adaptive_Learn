# 6-Week Implementation Plan: AI Educational Agent
**Target:** MVP completion in 1.5 Months | **API:** Groq (Free Tier) | **Framework:** LangChain

---

## Phase 1: The Core Engine (Backend First)

### **Week 1: Foundation & "The Brain"**
* **Focus:** Setting up the Groq Agent and basic document parsing.
* **Features to Build:**
    * **Environment Setup:** Python, LangChain, and Groq API (`Llama-3` or `Mixtral`) integration.
    * **Rate-Limit Handler:** A wrapper function for Groq’s Free Tier `429` errors using exponential backoff.
    * **MarkItDown Tool:** Integrate Microsoft’s `MarkItDown` library to convert PDFs/Docs into clean text.
    * **The "Hello World" Agent:** A LangChain agent that summarizes local text files.
* **End of Week Goal:** A Python script that takes a PDF path and prints a concise AI-generated summary to the terminal.

### **Week 2: Knowledge Retrieval (RAG)**
* **Focus:** Giving the Agent a "Memory" to save tokens and handle large files.
* **Features to Build:**
    * **Vector Database:** Set up **ChromaDB** (local/free) to store document embeddings.
    * **Chunking Strategy:** Split long textbooks into 1000-character pieces using `RecursiveCharacterTextSplitter`.
    * **Retrieval Tool:** A tool the Agent calls to "Search the document" for specific facts.
    * **Streamlit MVP:** A basic UI with a file uploader and a chat box for Q&A.
* **End of Week Goal:** A functional web app where you can chat with your PDF and get specific answers instead of just general summaries.

---

## Phase 2: Educational Tooling & Flashcards

### **Week 3: Assessment & Flashcard Generation**
* **Focus:** Automated material creation from processed text.
* **Features to Build:**
    * **Flashcard Tool:** A prompt template that identifies "Key Concepts" and "Definitions" to produce a JSON list of Front/Back cards.
    * **Quiz Tool:** Logic to generate MCQs and True/False questions in structured JSON.
    * **JSON Parser:** A robust function to ensure Groq's output is correctly formatted for the UI.
    * **Flashcard UI:** A simple "Card Flip" component in Streamlit using CSS.
* **End of Week Goal:** The Agent can take a chapter and generate a deck of 10 digital flashcards and a 5-question quiz.

### **Week 4: Audio & Multimedia Integration**
* **Focus:** Accessibility and auditory learning.
* **Features to Build:**
    * **Audio Summary Tool:** Integrate `edge-tts` (Free) to convert Agent summaries into MP3 files.
    * **Spoken Flashcards:** Add a "Play Audio" button to the flashcards for pronunciation/auditory study.
    * **YouTube Tool:** Use `youtube-transcript-api` to allow the Agent to summarize and quiz based on video URLs.
    * **Dashboard Polish:** Use Streamlit tabs to separate "Chat," "Flashcards," and "Quiz."
* **End of Week Goal:** A student can paste a YouTube link and receive both an audio lesson and a set of flashcards.

---

## Phase 3: Intelligence & Deployment

### **Week 5: Adaptive Logic & Spaced Repetition**
* **Focus:** Making the Agent "Smart" about student memory.
* **Features to Build:**
    * **Confidence Tracking:** Add "Mastered" vs. "Need Review" buttons to flashcards.
    * **Spaced Repetition (Lite):** A basic logic to re-show "Need Review" cards more frequently.
    * **Adaptive Tool:** The Agent analyzes missed quiz questions to generate a "Custom Remedial Study Guide."
    * **Analytics:** Visual charts (using Plotly) showing knowledge mastery over time.
* **End of Week Goal:** An intelligent study loop that tells the student exactly what to focus on today based on past performance.

### **Week 6: Polish, Test & Deploy**
* **Focus:** Production readiness and final presentation.
* **Features to Build:**
    * **Prompt Refinement:** Fine-tune the "Tutor Persona" for encouraging and accurate feedback.
    * **Error Resilience:** Add user-friendly messages for API outages or empty file uploads.
    * **Deployment:** Host the application on **Streamlit Cloud** (Free).
    * **Documentation:** Finalize `README.md` and record a 5-minute demo video.
* **End of Week Goal:** A live, shareable URL of your AI Educational Platform for your portfolio.

---

## Technical Tool Specifications (For Week 1 Development)

| Task | Tool/Library | Free Resource Detail |
| :--- | :--- | :--- |
| **LLM Inference** | `langchain-groq` | Groq Free Tier (high-speed inference) |
| **PDF Processing** | `markitdown` | Microsoft’s free MD converter |
| **Storage** | `chromadb` | Local persistent storage (no cloud cost) |
| **Voice** | `edge-tts` | Uses Microsoft Edge free TTS engine |
| **Frontend** | `streamlit` | Rapid UI development in Python |