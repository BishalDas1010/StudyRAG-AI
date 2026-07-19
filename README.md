# StudyRAG-AI

StudyRAG is an intelligent document-based learning assistant built for answering questions from uploaded PDF study materials. It supports both digital PDFs and scanned PDFs, uses OCR when needed, stores semantic embeddings in a vector database, and retrieves the most relevant context using Retrieval-Augmented Generation (RAG).

## Overview

This project helps users:
- upload a PDF study document
- process it into searchable chunks
- ask questions about the content
- get grounded answers with source references

It combines:
- PDF parsing and OCR pipelines
- vector search with ChromaDB
- LLM-powered answer generation
- a FastAPI backend
- a Streamlit frontend

## Features

- Supports both scanned and non-scanned PDFs
- OCR-based processing for image-heavy documents
- Semantic retrieval using embeddings
- Mistral-powered answer generation
- REST API for upload and querying
- Streamlit chat interface for easy interaction

## Project Structure

- `pdf_loader/` – main PDF processing, retrieval, and RAG logic
- `UI/` – Streamlit frontend for uploading PDFs and asking questions
- `chroma_db/` – local vector database storage
- `requirements.txt` – Python dependencies
- `Dockerfile` – container configuration for deployment

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- ChromaDB
- Mistral AI
- PyMuPDF / PyPDF
- Tesseract OCR

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables, including the required API key for the LLM provider.

## Running the Application

### Start the API backend

```bash
python pdf_loader/API.py
```

The backend will run on:
- `http://localhost:8000`

### Start the Streamlit UI

Open a new terminal and run:

```bash
streamlit run UI/home_page.py
```

## Usage

1. Open the Streamlit interface in your browser.
2. Upload a PDF file.
3. Click `Upload & Process`.
4. Ask questions about the uploaded document in the chat box.

## Notes

- The system automatically detects whether the PDF is scanned or digitally readable.
- Scanned PDFs use OCR before embedding and retrieval.
- The vector store is persisted locally in the `chroma_db/` folder.

## License

This project is intended for educational and research use.

