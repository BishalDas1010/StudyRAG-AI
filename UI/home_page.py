import streamlit as st
import requests
from datetime import datetime

# ---------- Configuration ----------
API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="PDF Q&A Chat",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for chat bubbles ----------
st.markdown("""
<style>
    .user-msg {
        background: #5585b5;
        color: #ffffff;
        padding: 10px 15px;
        border-radius: 18px 18px 5px 18px;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 10px;
        text-align: left;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .assistant-msg {
        background: #79c2d0;
        color: #1a1a1a;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        max-width: 80%;
        margin-right: auto;
        margin-bottom: 10px;
        text-align: left;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .msg-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .avatar {
        font-size: 28px;
        margin-right: 8px;
        margin-top: 4px;
        flex-shrink: 0;
    }
    .avatar-right {
        margin-left: 8px;
        margin-right: 0;
    }
    .source-box {
        background: #5585b5;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 6px;
        font-size: 0.9em;
    }
    .timestamp {
        font-size: 0.7em;
        color: #888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **RAG** system to answer questions about a PDF.
    - Vector store built at startup.
    - Answers with source references.
    """)
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    try:
        health = requests.get("http://localhost:8000/health")
        status = "✅ Connected" if health.status_code == 200 else "❌ Offline"
    except:
        status = "❌ Offline"
    st.caption(f"API Status: {status}")

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Main Chat Display ----------
st.title("📖 Ask Your PDF")
st.caption("Type a question below about the content of the PDF.")

# Display all messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-row" style="justify-content: flex-end;">
            <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <div class="user-msg">
                    {msg["content"]}
                </div>
                <div class="timestamp">{msg.get("time", "")}</div>
            </div>
            <div class="avatar avatar-right">🧑</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = "".join([
            f'<div class="source-box">📄 Page {src.get("page", "unknown")}: {src.get("content", "")[:150]}...</div>'
            for src in msg.get("sources", [])
        ])
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar">🤖</div>
            <div style="display: flex; flex-direction: column;">
                <div class="assistant-msg">
                    {msg["content"]}
                </div>
                <div class="timestamp">{msg.get("time", "")}</div>
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- Chat Input ----------
if prompt := st.chat_input("Ask something about the PDF..."):
    # Add user message
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": now
    })
    
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"query": prompt})
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "time": datetime.now().strftime("%I:%M %p")
                })
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI server. Make sure it's running at http://localhost:8000")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    
    # Force a rerun so the assistant's reply appears immediately
    st.rerun()