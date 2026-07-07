from PIL import Image
import pytesseract

image = Image.open(
    "/home/vishal/StudyRAG/pdf_loader/image.png"
)

text = pytesseract.image_to_string(image)

print(text)