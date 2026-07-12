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


load_dotenv()
    

if __name__ == "__main__":

    #path = "pdf_loader/image.png"

    path ="cs181-textbook.pdf"
    # Create pipeline objects
    digital_pipeline = Pipeline(path)
    ocr_pipeline = Ocr_main(path)

    # Detect PDF type
    is_scanned = digital_pipeline.detect_pdf_type()
    print(is_scanned)
    
    if is_scanned == False:
        ocr_text = ocr_pipeline.main()
        if not ocr_text:
            print("ocr_pipeline_faild")
        else:
            print("pipeline working")
        model = ocr_pipeline.Embadding_model()
        vactor_stor_db_ocr = ocr_pipeline.chroma_db_ocr(model,ocr_text)
        print("data store in chroma db")

        print("retriver ")


        search= Similarity_search(
            vector_db=vactor_stor_db_ocr,
            query="what is Machine learning",
            RETRIEVER_K=5,
            MMR_LAMBDA=0.5,
        )
        #compression Search 
        #LLm
        llm = ChatMistralAI(
            model = "mistral-large-latest"
        )

        Query ="What is Machine Learning?"
        retriver_result = search.contextualCompressionRetriever(
            llm=llm,
            query=Query
            )
        
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.3
        )
        for chunk ,doc in enumerate(retriver_result):
            print("+"*80)
            print(f"Result{1+chunk}")
            print(" ")
            print(doc.page_content)

        

    else:
        digital_pdf_loader = digital_pipeline.pdf_loader()
        digital_pdf_embadding  = digital_pipeline.embedding()
        vactor_stor_db_PDF= digital_pipeline.chroma_db(embedding_model=digital_pdf_embadding, docs=digital_pdf_loader)
        print("retriver ")

        print("retriver ")


        search = Similarity_search(
            vactor_stor_db_PDF,
            query="what is Machine learning",
            RETRIEVER_K=5,
            MMR_LAMBDA=0.5,
        )



        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.3
        )

        retriver_result = search.contextualCompressionRetriever(llm=llm,query="what is ML")

        for i, doc in enumerate(retriver_result):
            print("=" * 80)
            print(f"Result {i + 1}")
            print(doc.metadata)
            print(doc.page_content[:300])   # first 300 characters


        #LL
        genarator =RAGGenerator(llm)
        ans = genarator.generate(
            retriver_result,
            "what is ML"
        )
        print(ans.content)



        