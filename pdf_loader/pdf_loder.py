from digital_pipeline import Pipeline
from OCR_pipeline import Ocr_main


if __name__ == "__main__":

    path = "cs181-textbook.pdf"

    # Create pipeline objects
    digital_pipeline = Pipeline(path)
    ocr_pipeline = Ocr_main(path)

    # Detect PDF type
    is_scanned = digital_pipeline.detect_pdf_type()

    if is_scanned:

        print("Scanned PDF detected")

        chunks_pdf_ocr = ocr_pipeline.main()

        print(chunks_pdf_ocr)

    else:

        print("Digital PDF detected")

        chunks = digital_pipeline.pdf_loader()

        print(chunks)
        #model and chunk store in chDB
        model =digital_pipeline.embedding()
        digital_pipeline.chroma_db(model,chunks)

