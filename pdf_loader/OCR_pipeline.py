from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Ocr:

    def __init__(self, path):
        self.path = path

    def imagess(self, uploaded_images):

        documents = []

        for i, uploaded_image in enumerate(uploaded_images):

            # Open image safely
            with Image.open(uploaded_image) as image:

                # OCR
                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(uploaded_image),
                    "image_number": i + 1,
                    "type": "image",
                    "extraction_method": "ocr"
                }
            )

            documents.append(doc)

            print(
                f"OCR completed: {uploaded_image}"
            )

        return documents

    def pdff(self):

        images = convert_from_path(
            self.path,
            dpi=300
        )

        documents = []

        for page_number, image in enumerate(
            images,
            start=1
        ):

            print(
                f"Scanning page {page_number}"
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(self.path),
                    "page": page_number,
                    "type": "pdf",
                    "extraction_method": "ocr"
                }
            )

            documents.append(doc)

        return documents

    def chunking(self, docs):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        return chunks


class Ocr_main:

    def __init__(self, path_doco):
        self.path = path_doco

    def main(self):

        ocr_model = Ocr(self.path)

        # Get lowercase extension
        extension = Path(self.path).suffix.lower()

        if extension == ".pdf":

            print("PDF detected")

            pdf_documents = ocr_model.pdff()

            chunks = ocr_model.chunking(
                pdf_documents
            )

            return chunks

        elif extension in (
            ".jpg",
            ".jpeg",
            ".png"
        ):

            print("Image detected")

            image_documents = ocr_model.imagess(
                [self.path]
            )

            chunks = ocr_model.chunking(
                image_documents
            )

            return chunks

        else:

            print("Unsupported file")

            return []


if __name__ == "__main__":

    path = "/home/vishal/StudyRAG/pdf_loader/image.png"

    pipeline = Ocr_main(path)

    documents = pipeline.main()

    for doc in documents:

        print("\nMetadata:")
        print(doc.metadata)

        print("-" * 50)

        print("Content:")
        print(doc.page_content)