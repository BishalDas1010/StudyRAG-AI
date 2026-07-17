import streamlit as st
from pathlib import Path
import sys
import tempfile
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main
from Similarity import Similarity_search
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from RAGGenerator import RAGGenerator

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")

# ---------------- session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "is_scanned" not in st.session_state:
    st.session_state.is_scanned = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    st.subheader("Retriever Config")
    retriever_k = st.slider("Retriever K", 1, 15, 5)
    mmr_lambda = st.slider("MMR Lambda", 0.0, 1.0, 0.5, 0.05)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.05)

    process_btn = st.button("🚀 Process PDF", use_container_width=True, type="primary")

    if st.session_state.pdf_processed:
        st.success(f"✅ Processed: {st.session_state.pdf_name}")
        st.info("Type: " + ("Scanned (OCR)" if st.session_state.is_scanned is False else "Digital"))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📄 RAG PDF Chat Assistant")
st.caption("Upload a PDF, ask questions, get answers with sources")

# ---------------- process pdf ----------------
if process_btn:
    if uploaded_file is None:
        st.error("Upload a PDF first")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            digital_pipeline = Pipeline(tmp_path)
            ocr_pipeline = Ocr_main(tmp_path)

            is_scanned = digital_pipeline.detect_pdf_type()
            st.session_state.is_scanned = is_scanned

            if is_scanned is False:
                status = st.status("Running OCR pipeline...", expanded=True)
                status.write("Extracting text via OCR...")
                ocr_text = ocr_pipeline.main()

                if not ocr_text:
                    status.update(label="OCR failed", state="error")
                    st.error("OCR pipeline failed to extract text")
                else:
                    status.write("Building embeddings...")
                    model = ocr_pipeline.Embadding_model()
                    status.write("Storing in ChromaDB...")
                    vector_db = ocr_pipeline.chroma_db_ocr(model, ocr_text)

                    st.session_state.vector_db = vector_db
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    status.update(label="Done! (OCR pipeline)", state="complete")
            else:
                status = st.status("Running digital PDF pipeline...", expanded=True)
                status.write("Loading PDF...")
                docs = digital_pipeline.pdf_loader()
                status.write("Building embeddings...")
                embedding_model = digital_pipeline.embedding()
                status.write("Storing in ChromaDB...")
                vector_db = digital_pipeline.chroma_db(embedding_model=embedding_model, docs=docs)

                st.session_state.vector_db = vector_db
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                status.update(label="Done! (Digital pipeline)", state="complete")

            st.rerun()
        except Exception as e:
            st.error(f"Processing failed: {e}")
        finally:
            os.unlink(tmp_path)

# ---------------- chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Result {i + 1}**")
                    if doc.metadata:
                        st.caption(str(doc.metadata))
                    st.text(doc.page_content[:500])
                    st.divider()

# ---------------- chat input ----------------
if not st.session_state.pdf_processed:
    st.info("👆 Upload and process a PDF from the sidebar to start chatting")

query = st.chat_input(
    "Ask a question about the PDF...",
    disabled=not st.session_state.pdf_processed,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                llm_retriever = ChatMistralAI(model="mistral-large-latest")

                search = Similarity_search(
                    vector_db=st.session_state.vector_db,
                    query=query,
                    RETRIEVER_K=retriever_k,
                    MMR_LAMBDA=mmr_lambda,
                )
                retriever_result = search.contextualCompressionRetriever(
                    llm=llm_retriever,
                    query=query,
                )

                llm_gen = ChatMistralAI(model="mistral-large-latest", temperature=temperature)
                generator = RAGGenerator(llm_gen)
                answer = generator.generate(retriever_result, query)

                st.markdown(answer.content)

                with st.expander(f"📚 Sources ({len(retriever_result)})"):
                    for i, doc in enumerate(retriever_result):
                        st.markdown(f"**Result {i + 1}**")
                        if doc.metadata:
                            st.caption(str(doc.metadata))
                        st.text(doc.page_content[:500])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.content,
                    "sources": retriever_result,
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")