"""pytesseract is a Python library that 
acts as a wrapper for Google’s Tesseract-OCR
 (Optical Character Recognition) engine. """

from PIL import Image
import pytesseract
from langchain_core.documents import Document


def imagess(uploaded_imagess):
    docoments =[]
    for i,uploaded_image in enumerate(uploaded_imagess):
        #open uploaded image
        image = Image.open(uploaded_image)
        #run OCR to Extract the text 
        text = pytesseract.image_to_string(image)

        doc = Document(
            page_content=text,
            metadata={
                "source": uploaded_image.name,
                "image_number": i + 1,
                "type": "image",
                "extraction_method": "ocr"
            }
        )
        docoments.append(doc)

        print(
            f"OCR completed: {uploaded_image.name}"
        )
    return docoments

