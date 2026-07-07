from PIL import Image
import pytesseract



print("image detacted")
image = Image.open(
    "pdf_loader/image.png"
)

print("image path store ")
text = pytesseract.image_to_string(
    image,
    lang="eng"
)


print(text)