import streamlit as st
import requests
from datetime import datetime

# ---- Configuration ----
#API_BASE = "http://localhost:8000"
API_BASE = "http://13.201.122.171:8000"
UPLOAD_URL = f"{API_BASE}/upload-pdf/"
ASK_URL = f"{API_BASE}/ask"
HEALTH_URL = f"{API_BASE}/health"

st.set_page_config(
    page_title="PDF Q&A Chat",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS (same as before, keep it) ----
st.markdown("""
<style>
    .user-msg { background: #5585b5; color: #ffffff; padding: 10px 15px; border-radius: 18px 18px 5px 18px; max-width: 80%; margin-left: auto; margin-bottom: 10px; text-align: left; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .assistant-msg { background: #79c2d0; color: #1a1a1a; padding: 10px 15px; border-radius: 18px 18px 18px 5px; max-width: 80%; margin-right: auto; margin-bottom: 10px; text-align: left; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .msg-row { display: flex; align-items: flex-start; margin-bottom: 12px; }
    .avatar { font-size: 28px; margin-right: 8px; margin-top: 4px; flex-shrink: 0; }
    .avatar-right { margin-left: 8px; margin-right: 0; }
    .source-box { background: #e8f0fe; border-radius: 8px; padding: 8px 12px; margin-top: 6px; font-size: 0.9em; border-left: 3px solid #5585b5; }
    .timestamp { font-size: 0.7em; color: #888; margin-top: 4px; }
    .upload-success { color: green; font-weight: bold; }
    .upload-error { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---- Session State Initialisation ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None

# ---- Sidebar ----
with st.sidebar:
    st.header(" PDF Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload & Process", use_container_width=True):
            with st.spinner("Uploading and processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(UPLOAD_URL, files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_filename = data.get("filename")
                        st.session_state.messages = []   # clear old chat
                        st.success(f" {data.get('message')} ({data.get('pages', 0)} pages)")
                        st.rerun()
                    else:
                        st.error(f" Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f" Error: {e}")

    st.divider()
    st.header("ℹ About")
    st.markdown("""
    This app uses a **RAG** system to answer questions about a PDF.
    - Upload a PDF first.
    - Answers include source references.
    """)
    st.divider()
    if st.button(" Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    # Health check
    try:
        health = requests.get(HEALTH_URL)
        if health.status_code == 200:
            data = health.json()
            db_ready = data.get("vector_db_ready", False)
            status = f"Connected (PDF loaded: {db_ready})"
        else:
            status = " API unreachable"
    except:
        status = " Offline"
    st.caption(f"API Status: {status}")

# ---- Main Chat Display ----
st.title("📖 Ask Your PDF")
if st.session_state.pdf_loaded:
    st.caption(f"Currently using: **{st.session_state.pdf_filename}**")
else:
    st.warning(" Please upload a PDF using the sidebar first." )

# Display chat messages (same as before)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-row" style="justify-content: flex-end;">
            <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <div class="user-msg">{msg["content"]}</div>
                <div class="timestamp">{msg.get("time", "")}</div>
            </div>
            <div class="avatar avatar-right">🧑</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = "".join([
            f'<div class="source-box"> Page {src.get("page", "unknown")}: {src.get("content", "")[:150]}…</div>'
            for src in msg.get("sources", [])
        ])
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar"></div>
            <div style="display: flex; flex-direction: column;">
                <div class="assistant-msg">{msg["content"]}</div>
                <div class="timestamp">{msg.get("time", "")}</div>
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---- Chat Input (disabled if no PDF loaded) ----
if prompt := st.chat_input("Ask something about the PDF...", disabled=not st.session_state.pdf_loaded):
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": prompt, "time": now})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(ASK_URL, json={"query": prompt})
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
            st.error(" Cannot connect to FastAPI server. Make sure it's running at http://localhost:8000")
        except Exception as e:
            st.error(f" Error: {e}")

    st.rerun()