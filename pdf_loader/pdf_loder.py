from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("cs181-textbook.pdf")
documents = loader.load()

total_text = "".join(
    doc.page_content.strip()   # WHAT you want
    for doc in documents       # LOOP
)

if len(total_text) > 100:
    print("Digital PDF detected")
    print("Use PyMuPDFLoader")
else:
    print("Scanned PDF detected")
    print("Use OCR")