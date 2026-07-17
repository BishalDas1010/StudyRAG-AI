from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main
from Similarity import Similarity_search
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from RAGGenerator import RAGGenerator
import answer

load_dotenv()
if __name__ == "__main__":
    path = "cs181-textbook.pdf"
    
    # 1. Initialise pipelines
    digital_pipeline = Pipeline(path)
    ocr_pipeline = Ocr_main(path)
    
    # 2. Detect PDF type
    is_scanned = digital_pipeline.detect_pdf_type()
    print(f"Scanned: {is_scanned}")
    
    # 3. Build vector store and retriever
    if is_scanned == False:
        # OCR branch
        ocr_text = ocr_pipeline.main()
        if not ocr_text:
            print("OCR pipeline failed")
            sys.exit(1)
        model = ocr_pipeline.Embadding_model()
        vector_db = ocr_pipeline.chroma_db_ocr(model, ocr_text)
        print("Data stored in Chroma DB")
    else:
        # Digital PDF branch
        digital_pdf_loader = digital_pipeline.pdf_loader()
        digital_pdf_embadding = digital_pipeline.embedding()
        vector_db = digital_pipeline.chroma_db(digital_pdf_embadding, digital_pdf_loader)
        print("Data stored in Chroma DB")
    
    # 4. Set up LLM and retriever (common to both branches)
# After building vector_db and search object
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
    search = Similarity_search(
        vector_db=vector_db,
        query="dummy",               # placeholder, not used in retrieval
        RETRIEVER_K=5,
        MMR_LAMBDA=0.5,
    )

    print("\n Ready to answer questions about the PDF. Type 'exit' to quit.\n")

    while True:
        user_query = input(" ?? Your question: ")
        if user_query.lower() in ('exit', 'quit', 'q'):
            print("Goodbye!")
            break
        if not user_query.strip():
            continue

        # 1. Get compressed documents for this query
        docs = search.contextualCompressionRetriever(llm=llm, query=user_query)

        # 2. Generate answer from those documents
        generator = RAGGenerator(llm)
        answer = generator.generate(docs, user_query)

        print("\n" + "="*80)
        print(f"Q: {user_query}")
        print(f" A: {answer.content}")
        print("="*80 + "\n")