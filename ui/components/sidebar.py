import streamlit as st
import os
import uuid
from core.engine import process_document, create_tutor_agent
from tools.parser import text_to_md
from core.provider import get_llm
from tools.youtube_tool import get_youtube_transcript
from core.learner_service import log_activity
from database.crud import list_students, get_or_create_student
from ui.session_state import set_active_student, set_active_document, get_active_student_id, reset_document_artifacts
from ui.components.topic_badges import render_topic_badges
from ui.components.session_timer import show_session_timer

UPLOAD_FOLDER = "uploaded_docs"


def _student_selector():
    """Renders the student profile picker. Every quiz/flashcard event is attributed to this student_id."""
    show_synthetic = st.session_state.get("show_synthetic_students", False)
    students = list_students(include_synthetic=show_synthetic)
    choices = ["__new__"] + [s["id"] for s in students]
    labels = {s["id"]: s for s in students}

    current = st.session_state.get("student_id")
    if current and current not in choices:
        set_active_student(None)

    def label(student_id):
        if student_id == "__new__":
            return "➕ New Student"
        s = labels.get(student_id)
        if not s:
            return student_id
        return f"{s['name']}" + (f"  ·  {s['archetype']}" if s.get("archetype") else "")

    current = st.session_state.get("student_id")
    default_index = choices.index(current) if current in choices else 0

    selected = st.selectbox(
        "👤 Student Profile",
        choices,
        index=default_index,
        format_func=label,
        key="student_selector",
    )

    if selected == "__new__":
        new_name = st.text_input("New student name", key="new_student_name", placeholder="e.g. Alex")
        if st.button("Create Student", use_container_width=True):
            clean_name = new_name.strip()
            if clean_name:
                existing = next((s for s in students if s["name"].lower() == clean_name.lower()), None)
                if existing:
                    set_active_student(existing["id"])
                else:
                    new_id = "U" + uuid.uuid4().hex[:8].upper()
                    student = get_or_create_student(new_id, name=clean_name)
                    set_active_student(student["id"])
                st.rerun()
            else:
                st.warning("Enter a name first.")
    elif selected != current:
        set_active_student(selected)
        st.rerun()

    st.checkbox("🧪 Show synthetic test profiles", key="show_synthetic_students")


def show_sidebar():
    with st.sidebar:
        st.title("🎓 Veras AI Tutor")
        st.caption("Autonomous Agentic Academic Companion")

        if st.button("🔄 Reset / Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        _student_selector()
        if get_active_student_id():
            show_session_timer(get_active_student_id())

        # ----------------------------
        # Study Plan Quick Access
        # ----------------------------
        if st.session_state.get("study_plan"):
            st.divider()
            st.subheader("🗓️ 3-Day Study Roadmap")
            st.download_button(
                label="📥 Download Study Plan",
                data=st.session_state.study_plan,
                file_name="study_plan.md",
                mime="text/markdown",
                use_container_width=True
            )
            if st.button("💬 View Plan in Chat", use_container_width=True):
                log_activity(get_active_student_id(), "summary_read", st.session_state.get("document_id"))
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### 🗓️ Your Personalized Study Roadmap:\n\n{st.session_state.study_plan}"
                })
                st.rerun()

        st.divider()

        if st.session_state.get("topics"):
            render_topic_badges(st.session_state.topics, st.session_state.topic_chunk_counts)
            st.divider()

        st.subheader("📁 Ingest Knowledge")

        if not get_active_student_id():
            st.info("Select or create a student profile above before uploading material.")
            return st.session_state.get("current_file")

        input_method = st.radio("Select Input Source", ["PDF Document", "YouTube Lecture URL"])

        uploaded_file = None
        youtube_url = None

        if input_method == "PDF Document":
            uploaded_file = st.file_uploader(
                "Upload Course Material (PDF)",
                type=["pdf", "txt", "md"]
            )
        else:
            youtube_url = st.text_input("Enter YouTube Video URL")
            process_btn = st.button("Ingest Lecture Video", use_container_width=True)

        # ----------------------------
        # Process Uploaded PDF / Doc
        # ----------------------------
        if uploaded_file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

            if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Agent is ingesting, indexing, extracting topics, and analyzing document..."):
                    try:
                        llm = get_llm()
                        full_text = text_to_md(llm, file_path)
                        st.session_state.full_text = full_text

                        ingestion = process_document(file_path, student_id=get_active_student_id())
                        vector_db = ingestion["vector_db"]

                        agent_executor = create_tutor_agent(
                            vector_db=vector_db,
                            full_text=full_text,
                            doc_name=uploaded_file.name,
                            student_id=get_active_student_id(),
                            document_id=ingestion["document_id"]
                        )

                        # Agent autonomously builds initial study plan
                        agent_plan = agent_executor.run("Create a comprehensive 3-Day Mastery Plan for this document.")
                        plan_text = agent_plan.get("artifacts", {}).get("study_plan") or agent_plan.get("response", "")
                        st.session_state.study_plan = plan_text

                        st.session_state.vector_db = vector_db
                        st.session_state.agent_executor = agent_executor
                        st.session_state.current_file = uploaded_file.name
                        set_active_document(ingestion["document_id"], ingestion["topics"], ingestion["topic_chunk_counts"])
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"Hello! I have thoroughly ingested and indexed **{uploaded_file.name}**. I found {len(ingestion['topics'])} core topics and prepared a customized 3-Day Study Mastery Plan for you. Feel free to ask questions, or ask me to generate flashcards, practice quizzes, audio summaries, or explain complex concepts!"
                        }]
                        reset_document_artifacts()

                        st.success(f"Ready to study: {uploaded_file.name}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error initializing agent: {e}")

        # ----------------------------
        # Process YouTube Link
        # ----------------------------
        elif youtube_url and (
            (input_method == "YouTube Lecture URL" and "current_file" not in st.session_state)
            or st.session_state.get("current_file") != youtube_url
        ):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(UPLOAD_FOLDER, "youtube_transcript.txt")

            with st.spinner("Fetching YouTube transcript and deploying Agent..."):
                try:
                    transcript = get_youtube_transcript(youtube_url)

                    if transcript.startswith("Error"):
                        st.error(transcript)
                    else:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(transcript)

                        llm = get_llm()
                        full_text = text_to_md(llm, file_path)
                        st.session_state.full_text = full_text

                        ingestion = process_document(file_path, student_id=get_active_student_id())
                        vector_db = ingestion["vector_db"]

                        agent_executor = create_tutor_agent(
                            vector_db=vector_db,
                            full_text=full_text,
                            doc_name=f"YouTube Lecture ({youtube_url})",
                            student_id=get_active_student_id(),
                            document_id=ingestion["document_id"]
                        )

                        # Agent autonomously builds initial study plan
                        agent_plan = agent_executor.run("Create a comprehensive 3-Day Mastery Plan for this lecture transcript.")
                        plan_text = agent_plan.get("artifacts", {}).get("study_plan") or agent_plan.get("response", "")
                        st.session_state.study_plan = plan_text

                        st.session_state.vector_db = vector_db
                        st.session_state.agent_executor = agent_executor
                        st.session_state.current_file = youtube_url
                        set_active_document(ingestion["document_id"], ingestion["topics"], ingestion["topic_chunk_counts"])
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"Hello! I have transcribed and indexed the YouTube lecture, and found {len(ingestion['topics'])} core topics. I've prepared a 3-Day Study Mastery Plan. What concept would you like to explore first?"
                        }]
                        reset_document_artifacts()

                        st.success("YouTube lecture processed and ready!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error processing YouTube video: {e}")

        return st.session_state.get("current_file")
