import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from contextlib import asynccontextmanager
import asyncio

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main
from Similarity import Similarity_search
from RAGGenerator import RAGGenerator

load_dotenv()

# Global objects
vector_db = None
llm = None
search = None
upload_lock = asyncio.Lock()          # prevent concurrent uploads
CURRENT_PDF_PATH = None              # track the currently active PDF (optional)

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

# Lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, llm, search
    # Optionally build a default vector store if you have a default PDF.
    # For this fix, we skip building at startup and require an upload first.
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
    print("LLM initialised. Upload a PDF to start asking questions.")
    yield
    print("Shutting down...")

app = FastAPI(
    title="PDF RAG API",
    description="Ask questions about a PDF. Upload a PDF first.",
    lifespan=lifespan
)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db, search, CURRENT_PDF_PATH

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Acquire lock to prevent concurrent rebuilds
    async with upload_lock:
        # Save the uploaded file to a temporary location
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        finally:
            await file.close()

        # Build the vector store from the temporary PDF
        try:
            db = build_vector_store(tmp_path)
            vector_db = db
            CURRENT_PDF_PATH = tmp_path   # keep track for possible cleanup later
            # Reinitialise the search object with the new vector store
            search = Similarity_search(
                vector_db=vector_db,
                query="dummy",           # not used for retrieval
                RETRIEVER_K=5,
                MMR_LAMBDA=0.5,
            )
        except Exception as e:
            # Clean up temporary file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "message": "PDF uploaded and processed successfully",
        "filename": file.filename,
        "pages": len(vector_db.get()["ids"]) if vector_db else 0   # rough page count
    }

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]   # each with page, content snippet

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if not search or not llm or not vector_db:
        raise HTTPException(status_code=503, detail="No PDF has been uploaded yet. Please upload a PDF first.")
    
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
        "API:app",         
        host="0.0.0.0",
        port=8000,
        reload=True         # enables auto‑reload during development
    )