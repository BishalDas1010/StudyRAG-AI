from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from contextlib import asynccontextmanager

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main
# Ocr is imported but not used; you can remove it if not needed
from Similarity import Similarity_search
from RAGGenerator import RAGGenerator

load_dotenv()

# ---------- Global objects ----------
vector_db = None
llm = None
search = None

def build_vector_store(pdf_path: str):
    """Build the vector store based on PDF type."""
    digital_pipeline = Pipeline(pdf_path)
    ocr_pipeline = Ocr_main(pdf_path)
    
    is_scanned = digital_pipeline.detect_pdf_type()  # returns True or False
    print(f"PDF scanned: {is_scanned}")
    
    if not is_scanned:
        ocr_text = ocr_pipeline.main()
        if not ocr_text:
            raise RuntimeError("OCR pipeline failed")
        model = ocr_pipeline.Embadding_model()
        db = ocr_pipeline.chroma_db_ocr(model, ocr_text)
    else:
        loader = digital_pipeline.pdf_loader()
        embedding = digital_pipeline.embedding()
        db = digital_pipeline.chroma_db(embedding, loader)
    
    print("Vector store built successfully.")
    return db

# ---------- Lifespan (startup/shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build vector store and initialise LLM, search
    global vector_db, llm, search
    pdf_path = "cs181-textbook.pdf"   # change as needed
    try:
        vector_db = build_vector_store(pdf_path)
        llm = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
        search = Similarity_search(
            vector_db=vector_db,
            query="dummy",           # not used for retrieval
            RETRIEVER_K=5,
            MMR_LAMBDA=0.5,
        )
        print("Ready to answer questions.")
    except Exception as e:
        print(f" Startup error: {e}")
        # Depending on your use case, you might want to raise or exit
        raise RuntimeError("Failed to initialise the RAG system") from e
    
    yield   # The application runs here
    
    # Shutdown: (optional) cleanup resources if needed
    print("Shutting down...")

# ---------- FastAPI app ----------
app = FastAPI(
    title="PDF RAG API",
    description="Ask questions about the PDF.",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]   # each with page, content snippet

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if not search or not llm or not vector_db:
        raise HTTPException(status_code=503, detail="System not initialised")
    
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Retrieve compressed documents
    docs = search.contextualCompressionRetriever(llm=llm, query=query)
    
    # Generate answer
    generator = RAGGenerator(llm)
    answer_obj = generator.generate(docs, query)
    
    # Prepare sources
    sources = []
    for doc in docs:
        sources.append({
            "page": doc.metadata.get("page", "unknown"),
            "content": doc.page_content[:200]   # short snippet
        })
    
    return QueryResponse(answer=answer_obj.content, sources=sources)

@app.get("/health")
async def health():
    return {"status": "ok", "vector_db_ready": vector_db is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "API:app",          # or "app:app" if the file is named app.py
        host="0.0.0.0",
        port=8000,
        reload=True         # enables auto‑reload during development
    )