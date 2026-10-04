import os
import json
from typing import List, Dict, Any, Optional

from core.provider import get_llm
from core.vector_store import store_chunks
from core.topic_extractor import extract_and_tag_document
from prompts.system_prompt import TUTOR_SYSTEM_PROMPT
from tools.parser import text_to_md
from tools.chunker import chunk_text
from tools.retrieval_tool import search_document, create_retrieval_tool
from tools.flashcard_tool import generate_flashcards
from tools.quiz_tool import generate_quiz
from tools.planner_tool import generate_study_plan
from tools.adaptive_tool import generate_remedial_guide
from tools.audio_tool import create_audio_narration
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


class AgentArtifactStore:
    """
    Thread-safe container tracking artifacts generated during an agent session.
    """
    def __init__(self):
        self.flashcards: Optional[List[Dict[str, Any]]] = None
        self.quiz: Optional[List[Dict[str, Any]]] = None
        self.study_plan: Optional[str] = None
        self.remedial_guide: Optional[str] = None
        self.audio_file: Optional[str] = None

    def reset_recent(self):
        self.flashcards = None
        self.quiz = None
        self.study_plan = None
        self.remedial_guide = None
        self.audio_file = None

    def get_artifacts(self) -> Dict[str, Any]:
        return {
            "flashcards": self.flashcards,
            "quiz": self.quiz,
            "study_plan": self.study_plan,
            "remedial_guide": self.remedial_guide,
            "audio_file": self.audio_file
        }


class TutorAgent:
    """
    Autonomous Educational Agent orchestrating RAG search, flashcards,
    quizzes, study plans, adaptive remediation, and audio synthesis.
    """
    def __init__(self, vector_db=None, full_text: str = "", doc_name: str = "Document"):
        self.vector_db = vector_db
        self.full_text = full_text
        self.doc_name = doc_name
        self.artifacts = AgentArtifactStore()
        self.llm = get_llm()
        self.tools = self._build_tools()
        self.model_with_tools = self.llm.bind_tools(self.tools)

    def _get_context(self, query: str) -> str:
        """Helper to get contextual text via RAG search or full_text."""
        if self.vector_db is not None:
            retrieved = search_document(query, self.vector_db)
            if retrieved and not retrieved.startswith("No relevant"):
                return retrieved
        return self.full_text if self.full_text else query

    def _build_tools(self):
        artifacts = self.artifacts
        get_context = self._get_context
        doc_name = self.doc_name

        @tool
        def document_search(query: str) -> str:
            """Search the uploaded document for relevant facts, definitions, and conceptual context."""
            return get_context(query)

        @tool
        def flashcard_creator(topic: str = "General", num_cards: int = 10) -> str:
            """Generate active-recall study flashcards on a specific topic or key concepts from the document."""
            context = get_context(topic)
            cards = generate_flashcards(context, topic=topic, num_cards=num_cards)
            if cards:
                artifacts.flashcards = cards
                return f"Successfully generated {len(cards)} flashcards on '{topic}'. Cards are now available for study."
            return f"Could not generate flashcards for '{topic}' from the available context."

        @tool
        def quiz_creator(topic: str = "General", num_questions: int = 5) -> str:
            """Generate a practice assessment quiz with multiple-choice and True/False questions based on the document."""
            context = get_context(topic)
            quiz = generate_quiz(context, topic=topic, num_questions=num_questions)
            if quiz:
                artifacts.quiz = quiz
                return f"Successfully created a {len(quiz)}-question practice quiz on '{topic}'. The quiz is ready."
            return f"Could not generate a quiz for '{topic}' from the available context."

        @tool
        def study_planner(topic: str = "") -> str:
            """Generate or update a structured 3-day study mastery plan for the document or specified topic."""
            t_name = topic if topic else doc_name
            context = get_context(t_name)
            plan = generate_study_plan(context, doc_name=t_name)
            if plan:
                artifacts.study_plan = plan
                return f"Successfully created a 3-Day Mastery Plan for '{t_name}'.\n\n{plan}"
            return f"Could not generate a study plan for '{t_name}'."

        @tool
        def adaptive_remedial_evaluator(missed_questions_summary: str, topic: str = "General") -> str:
            """Analyze student errors and missed quiz questions to generate a personalized diagnostic remedial study guide."""
            context = get_context(topic)
            guide = generate_remedial_guide(missed_questions_summary, context=context)
            if guide:
                artifacts.remedial_guide = guide
                return f"Successfully generated a personalized Remedial Study Guide for the missed questions.\n\n{guide}"
            return "Could not generate a remedial guide."

        @tool
        def audio_narrator(text: str, filename_prefix: str = "summary") -> str:
            """Convert text summaries, study guides, or explanations into spoken audio speech (.mp3)."""
            audio_path = create_audio_narration(text, filename_prefix=filename_prefix)
            if not audio_path.startswith("Error"):
                artifacts.audio_file = audio_path
                return f"Audio narration generated successfully at: {audio_path}"
            return f"Failed to generate audio: {audio_path}"

        return [
            document_search,
            flashcard_creator,
            quiz_creator,
            study_planner,
            adaptive_remedial_evaluator,
            audio_narrator
        ]

    def _fallback_tool_for_query(self, query: str) -> Optional[str]:
        """Heuristic fallback when the model answers in prose instead of calling a tool."""
        normalized = (query or "").lower()
        if any(term in normalized for term in ["flashcard", "flashcards", "card deck", "study cards"]):
            return "flashcard_creator"
        if any(term in normalized for term in ["quiz", "assessment", "practice test", "mcq", "true/false"]):
            return "quiz_creator"
        if any(term in normalized for term in ["study plan", "3-day", "mastery plan", "roadmap"]):
            return "study_planner"
        if any(term in normalized for term in ["audio", "narrate", "read aloud", "tts", "voice"]):
            return "audio_narrator"
        if any(term in normalized for term in ["remedial", "missed question", "diagnose", "why was i wrong"]):
            return "adaptive_remedial_evaluator"
        return None

    def run(self, query: str, chat_history: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """
        Executes an autonomous ReAct tool-calling reasoning loop.
        """
        self.artifacts.reset_recent()

        messages = [SystemMessage(content=TUTOR_SYSTEM_PROMPT)]

        if chat_history:
            for role, content in chat_history:
                if role in ("human", "user"):
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))

        tools_by_name = {t.name: t for t in self.tools}

        # Autonomous reasoning loop (max 5 tool call iterations)
        for _ in range(5):
            ai_msg = self.model_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
                fallback_tool = self._fallback_tool_for_query(query)
                if fallback_tool and fallback_tool in tools_by_name:
                    selected_tool = tools_by_name[fallback_tool]
                    tool_args = {}
                    if fallback_tool == "flashcard_creator":
                        tool_args = {"topic": "General", "num_cards": 10}
                    elif fallback_tool == "quiz_creator":
                        tool_args = {"topic": "General", "num_questions": 5}
                    elif fallback_tool == "study_planner":
                        tool_args = {"topic": "General"}
                    elif fallback_tool == "audio_narrator":
                        tool_args = {"text": query, "filename_prefix": "summary"}
                    elif fallback_tool == "adaptive_remedial_evaluator":
                        tool_args = {"missed_questions_summary": query, "topic": "General"}

                    try:
                        tool_output = selected_tool.invoke(tool_args)
                    except Exception as e:
                        tool_output = f"Error executing {fallback_tool}: {str(e)}"
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=fallback_tool, name=fallback_tool))
                    break
                break

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", tool_name)

                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    try:
                        tool_output = selected_tool.invoke(tool_args)
                    except Exception as e:
                        tool_output = f"Error executing {tool_name}: {str(e)}"
                else:
                    tool_output = f"Tool '{tool_name}' is not recognized."

                messages.append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_id,
                    name=tool_name
                ))

        final_response = messages[-1].content if messages else "I have completed processing your request."
        
        return {
            "response": final_response,
            "artifacts": self.artifacts.get_artifacts(),
            "messages": messages
        }


def _make_document_id(file_path: str) -> str:
    """Deterministic short id so re-ingesting the same path updates, not duplicates, its topics."""
    import hashlib
    return "D" + hashlib.md5(file_path.encode("utf-8")).hexdigest()[:10]


def process_document(file_path: str, student_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingests and parses a document file: extracts a canonical topic taxonomy
    (1 LLM call), tags every chunk with its nearest topic via local cosine
    similarity, stores the chunks + topic_id metadata in ChromaDB, and
    persists the document, topic, and chunk rows to SQLite.
    """
    llm = get_llm()
    raw_markdown = text_to_md(llm, file_path)
    chunks = chunk_text(raw_markdown)
    doc_name = os.path.basename(file_path)
    document_id = _make_document_id(file_path)

    extraction = extract_and_tag_document(raw_markdown, chunks, llm=llm)
    raw_topics = extraction["topics"]
    tagged_chunks = extraction["tagged_chunks"]

    # 1. Initialize SQLite and persist Document + Topics first so foreign keys exist
    saved_topics = []
    try:
        from database.db import init_db
        from database.crud import create_document, replace_topics, replace_chunks
        init_db()
        create_document(document_id, doc_name, source_type="pdf", uploaded_by=student_id)
        if raw_topics:
            saved_topics = replace_topics(document_id, raw_topics)
    except Exception as e:
        print(f"Warning: could not persist document/topics to SQLite: {e}")

    # 2. Build aligned metadata, chunk rows, and dual-keyed chunk counts
    # Aligns topic_id format:
    # - full_topic_id (e.g. "D1234567890_T1") matches Topic.id primary key in SQLite & FK in Chunk
    # - topic_key (e.g. "T1") provides the short sequential label for prompts and UI badges
    # - chroma_id (e.g. "D1234567890_c0") deterministically links Chroma vectors to SQLite chunks
    metadatas = []
    chunks_data = []
    ids = []
    topic_chunk_counts = {}

    for i, tc in enumerate(tagged_chunks):
        short_key = tc.get("topic_id")
        full_topic_id = f"{document_id}_{short_key}" if short_key else None
        chroma_id = f"{document_id}_c{i}"
        ids.append(chroma_id)

        metadatas.append({
            "document_id": document_id,
            "topic_id": full_topic_id or "",
            "topic_key": short_key or "",
            "chunk_index": i,
        })

        chunks_data.append({
            "chunk_index": i,
            "topic_id": full_topic_id,
            "chroma_id": chroma_id,
            "text_preview": tc.get("chunk", "")[:300],
            "similarity": tc.get("similarity"),
        })

        if short_key:
            topic_chunk_counts[short_key] = topic_chunk_counts.get(short_key, 0) + 1
            if full_topic_id:
                topic_chunk_counts[full_topic_id] = topic_chunk_counts.get(full_topic_id, 0) + 1

    # 3. Persist chunk records to SQLite
    try:
        from database.crud import replace_chunks
        replace_chunks(document_id, chunks_data)
    except Exception as e:
        print(f"Warning: could not persist chunks to SQLite: {e}")

    # 4. Persist chunks to ChromaDB with deterministic IDs and aligned topic_id metadata
    vector_db = store_chunks(chunks, doc_name, metadatas=metadatas, ids=ids)

    return {
        "vector_db": vector_db,
        "document_id": document_id,
        "topics": saved_topics if saved_topics else raw_topics,
        "topic_chunk_counts": topic_chunk_counts,
        "chunk_count": len(chunks),
    }


def create_tutor_agent(vector_db=None, full_text: str = "", doc_name: str = "Document") -> TutorAgent:
    """
    Instantiates and returns an autonomous TutorAgent.
    """
    return TutorAgent(vector_db=vector_db, full_text=full_text, doc_name=doc_name)


def run_agent_query(agent: Any, query: str, chat_history=None) -> Any:
    """
    Unified execution wrapper returning both response string and capturing artifacts.
    """
    if isinstance(agent, TutorAgent):
        result = agent.run(query, chat_history=chat_history)
        return result
    else:
        # Fallback for generic agents
        return {"response": str(agent), "artifacts": {}}
