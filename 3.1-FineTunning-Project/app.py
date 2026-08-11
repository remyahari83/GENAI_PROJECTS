import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from llm_providers import run_llm

st.set_page_config(page_title="My First AI Chat Assistant", page_icon="🤖")
st.title("🤖 AI Chat Assistant")

# Initialize conversation memory
if "history" not in st.session_state:
    st.session_state["history"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat history
for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["history"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Run chosen provider
    reply = run_llm(st.session_state["history"])

    # Add assistant response
    st.session_state["history"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)

# Reset conversation
if st.button("🔄 Reset Conversation"):
    st.session_state["history"] = [{"role": "assistant", "content": "Conversation reset. How can I assist you?"}]
    st.rerun()
