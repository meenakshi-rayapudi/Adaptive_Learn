import streamlit as st
import os
from core.engine import process_document, create_tutor_agent
from tools.parser import text_to_md
from core.provider import get_llm
from tools.youtube_tool import get_youtube_transcript

UPLOAD_FOLDER = "uploaded_docs"

def show_sidebar():
    with st.sidebar:
        st.title("🎓 Veras AI Tutor")
        st.caption("Autonomous Agentic Academic Companion")

        if st.button("🔄 Reset / Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

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
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### 🗓️ Your Personalized Study Roadmap:\n\n{st.session_state.study_plan}"
                })
                st.rerun()

        st.divider()
        st.subheader("📁 Ingest Knowledge")
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

                with st.spinner("Agent is ingesting, indexing, and analyzing document..."):
                    try:
                        llm = get_llm()
                        full_text = text_to_md(llm, file_path)
                        st.session_state.full_text = full_text

                        vector_db = process_document(file_path)
                        agent_executor = create_tutor_agent(
                            vector_db=vector_db,
                            full_text=full_text,
                            doc_name=uploaded_file.name
                        )

                        # Agent autonomously builds initial study plan
                        agent_plan = agent_executor.run("Create a comprehensive 3-Day Mastery Plan for this document.")
                        plan_text = agent_plan.get("artifacts", {}).get("study_plan") or agent_plan.get("response", "")
                        st.session_state.study_plan = plan_text

                        st.session_state.vector_db = vector_db
                        st.session_state.agent_executor = agent_executor
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"Hello! I have thoroughly ingested and indexed **{uploaded_file.name}**. I've prepared a customized 3-Day Study Mastery Plan for you. Feel free to ask questions, or ask me to generate flashcards, practice quizzes, audio summaries, or explain complex concepts!"
                        }]
                        st.session_state.flashcards = None
                        st.session_state.quiz = None
                        st.session_state.remedial_guide = None

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

                        vector_db = process_document(file_path)
                        agent_executor = create_tutor_agent(
                            vector_db=vector_db,
                            full_text=full_text,
                            doc_name=f"YouTube Lecture ({youtube_url})"
                        )

                        # Agent autonomously builds initial study plan
                        agent_plan = agent_executor.run("Create a comprehensive 3-Day Mastery Plan for this lecture transcript.")
                        plan_text = agent_plan.get("artifacts", {}).get("study_plan") or agent_plan.get("response", "")
                        st.session_state.study_plan = plan_text

                        st.session_state.vector_db = vector_db
                        st.session_state.agent_executor = agent_executor
                        st.session_state.current_file = youtube_url
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": f"Hello! I have transcribed and indexed the YouTube lecture. I've prepared a 3-Day Study Mastery Plan. What concept would you like to explore first?"
                        }]
                        st.session_state.flashcards = None
                        st.session_state.quiz = None
                        st.session_state.remedial_guide = None

                        st.success("YouTube lecture processed and ready!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error processing YouTube video: {e}")

        return st.session_state.get("current_file")