
from digital_pipeline import Pipeline
if __name__ == "__main__":

    path = "cs181-textbook.pdf"

    d1 = Pipeline(path)

    if d1.detect_pdf_type():

        chunks = d1.pdf_loader()

        embedding_model = d1.embedding()

        vector_store = d1.chroma_db(
            embedding_model,
            chunks
        )

    else:
        print("OCR pipeline required")