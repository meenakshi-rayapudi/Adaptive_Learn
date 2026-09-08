## Week 1 Action Plan: "The Foundation"

### Objective: One-Click Document Summarization
By the end of this week, we will have a working Python backend that can ingest a PDF and output a high-quality summary via Groq.

---

### Team Assignments

#### 1. AI Infrastructure (Member 1)
- [ ] Install `langchain-groq` and `python-dotenv`.
- [ ] Create a secure `config.py` to load Groq API keys.
- [ ] Implement a basic Chat wrapper that handles the "Free Tier" limits.

#### 2. Document Parsing (Member 2)
- [ ] Install `markitdown`.
- [ ] Create a script to convert PDF/DOCX to Markdown.
- [ ] Ensure tables and headings are preserved in the text output.

#### 3. Agent Intelligence (Member 3)
- [ ] Design the System Prompt for the "AI Tutor."
- [ ] Create a `SummarizationChain` that limits token usage (to stay in Groq's free tier).
- [ ] Test prompts to ensure the AI doesn't hallucinate.

#### 4. System Flow (Member 4)
- [ ] Set up the project folder structure.
- [ ] Build the `main.py` "Orchestrator" that links parsing -> prompting -> output.
- [ ] Create a `requirements.txt` for the whole team.