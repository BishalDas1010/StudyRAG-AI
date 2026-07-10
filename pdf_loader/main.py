from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main
from Similarity_search import Similarity_search


if __name__ == "__main__":

    path = "pdf_loader/image.png"

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
        retiver = Similarity_search(vactor_stor_db_ocr,RETRIEVER_K =5,MMR_LAMBDA =1)
        

        

    else:
        digital_pdf_loader = digital_pipeline.pdf_loader()
        digital_pdf_embadding  = digital_pipeline.embedding()
        digital_pipeline.chroma_db(embedding_model=digital_pdf_embadding, docs=digital_pdf_loader)
        