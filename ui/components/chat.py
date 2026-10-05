import streamlit as st
from core.engine import run_agent_query
from core.learner_service import log_activity

def chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if agent is ready
    if "agent_executor" not in st.session_state:
        st.info("Please upload a study document or enter a YouTube lecture link to activate the AI Tutor.")
        return None

    # React to user input
    if prompt := st.chat_input("Ask a question, request flashcards, a practice quiz, study plan, or audio..."):
        # Display user message in chat message container
        log_activity(st.session_state.get("student_id"), "chat_question", st.session_state.get("document_id"))
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("AI Tutor is reasoning and analyzing tools..."):
                try:
                    history = []
                    for m in st.session_state.messages[:-1]: 
                        role = "human" if m["role"] == "user" else "ai"
                        history.append((role, m["content"]))

                    result = run_agent_query(st.session_state.agent_executor, prompt, history)
                    
                    if isinstance(result, dict):
                        response_text = result.get("response", "")
                        artifacts = result.get("artifacts", {})
                        
                        # Sync agent-generated artifacts to UI state
                        if artifacts.get("flashcards"):
                            st.session_state.flashcards = artifacts["flashcards"]
                        if artifacts.get("quiz"):
                            st.session_state.quiz = artifacts["quiz"]
                        if artifacts.get("study_plan"):
                            st.session_state.study_plan = artifacts["study_plan"]
                        if artifacts.get("remedial_guide"):
                            st.session_state.remedial_guide = artifacts["remedial_guide"]
                        if artifacts.get("audio_file"):
                            st.session_state.latest_audio = artifacts["audio_file"]
                    else:
                        response_text = str(result)

                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing agent query: {e}")
                    return None

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        return st.session_state.messages[-1]["content"]
    
    return None
