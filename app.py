import streamlit as st
from agents import run_single_agent, run_multi_agent

st.set_page_config(page_title="Financial Agent Chat", layout="wide")

st.title("MiniProject 3 Financial Chat")

# ===============================
# Session memory
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# Sidebar
# ===============================

with st.sidebar:

    st.header("Controls")

    architecture = st.selectbox(
        "Agent selector",
        ["Single Agent", "Multi-Agent"]
    )

    model = st.selectbox(
        "Model selector",
        ["gpt-4o-mini", "gpt-4o"]
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ===============================
# Display conversation
# ===============================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant":
            st.markdown(
                f"**Architecture:** {msg['architecture']}  \n"
                f"**Model:** {msg['model']}"
            )

        st.markdown(msg["content"])

# ===============================
# Chat input
# ===============================

user_input = st.chat_input("Ask a financial question")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    if architecture == "Single Agent":
        answer = run_single_agent(user_input, model, history)
    else:
        answer = run_multi_agent(user_input, model, history)

    with st.chat_message("assistant"):

        st.markdown(
            f"**Architecture:** {architecture}  \n"
            f"**Model:** {model}"
        )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "architecture": architecture,
        "model": model
    })