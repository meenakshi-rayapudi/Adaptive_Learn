import streamlit as st

def show_header():

    st.markdown(
        "<div class='main-title'>Educational AI Agent</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Upload a document and ask questions instantly</div>",
        unsafe_allow_html=True
    )