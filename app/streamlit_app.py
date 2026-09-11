import sys, os
import streamlit as st
import json, uuid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from modules.loader import load_document
from modules.splitter import split_documents
from modules.embedder import build_or_update_vectorstore
from modules.retriever import get_retriever
from modules.rag_chain import build_rag_chain
from modules.memory import add_memory_to_chain

# define page config
st.set_page_config(page_title = "Capstone RAG Chatbot", layout = "wide")

# load config
CONFIG_PATH = os.path.join(project_root, "config", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)


# session & multichat management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "active_session" not in st.session_state:
    st.session_state.active_session = None


# sidebar: chat sessions
st.sidebar.header("☸️ Chat Settings")

# new chat button
if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "name": f"Chat {len(st.session_state.chat_sessions) + 1}",
        "llm_provider": "openai",
        "memory_enabled": True,
        "chat_history": []
    }
    st.session_state.active_session = new_id

# if no session exists, initialize first chat
if not st.session_state.chat_sessions:
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "name": f"Chat 1",
        "llm_provider": "openai",
        "memory_enabled": True,
        "chat_history": []        
    }
    st.session_state.active_session = new_id


# select active chat session
session_names = {
    sid: info["name"] for sid, info in st.session_state.chat_sessions.items()
}

selected_name = st.sidebar.selectbox(
    "Active Chat Session",
    options=list(session_names.values()),
    index=list(session_names.values()).index(
        st.session_state.chat_sessions[st.session_state.active_session]["name"]
    ),
)


for sid, info in st.session_state.chat_sessions.items():
    if info["name"] == selected_name:
        st.session_state.active_session = sid
        break

current_session = st.session_state.chat_sessions[st.session_state.active_session]

# show session ID for reference: optional
st.sidebar.text(f"Session ID: {st.session_state.active_session}")

# session-specific LLM provider
current_session["llm_provider"] = st.sidebar.selectbox(
    "LLM Provider",
    ["openai", "gemini"],
    index = 0 if current_session["llm_provider"] == "openai" else 1,
    key = f"provider_{st.session_state.active_session}",
)

# session-specific memory toggle
current_session["memory_enabled"] = st.sidebar.toggle(
    "Enable Memory",
    value = current_session["memory_enabled"],
    key = f"memory_{st.session_state.active_session}"
)


# upload documents section
st.sidebar.subheader("📃 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF/DOCX/TXT", type=["pdf", "docx", "docs", "txt"]
)


# vectorstore path
os.path.join(project_root, "vectorstores", "faiss_index")

# handle the uploaded document
if uploaded_file:
    with st.spinner("Processing uploaded document..."):
        upload_path = os.path.join(
            project_root, "data", "uploaded_docs", uploaded_file.name
        )
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_document(upload_path)
        chunks = split_documents(docs, chunk_size=400, chunk_overlap=50)
        vectorstore = build_or_update_vectorstore(chunks, CONFIG)

        st.success(f"{uploaded_file.name}: processed and added to vectorstore!!!")


# build RAG chain for Active-Session

try:
    vectorstore = build_or_update_vectorstore([], CONFIG)
    retriever = get_retriever(vectorstore, CONFIG)

    CONFIG["llm_provider"] = current_session["llm_provider"]

    rag_chain = build_rag_chain(retriever, CONFIG)

    rag_chain_with_memory = add_memory_to_chain(
        rag_chain,
        session_id = st.session_state.active_session,
        enabled = current_session["memory_enabled"],
    )
except Exception as e:
    st.error(f"Error initializing backend: {e}")
    st.stop()


# -------------------------------------------------------------
# Chat UI Section
# -------------------------------------------------------------
st.title("💬 Capstone RAG Chatbot")

# --- Display chat messages for current session ---
for msg in current_session["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask your question..."):
    current_session["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            response = rag_chain_with_memory.invoke(
                {"question": prompt},
                config={
                    "configurable": {"session_id": st.session_state.active_session}
                },
            )
            full_response = getattr(response, "content", str(response))
        except Exception as e:
            full_response = f"⚠️ Error: {e}"

        response_placeholder.markdown(full_response)

    current_session["chat_history"].append(
        {"role": "assistant", "content": full_response}
    )