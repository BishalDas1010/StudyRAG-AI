from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyMuPDFLoader("cs181-textbook.pdf")
documents = loader.load()
path ="cs181-textbook.pdf"
total_text = "".join(
    doc.page_content.strip()
    for doc in documents
)


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main(path):
    # Create PDF loader object
    loader = PyPDFLoader(path)

    # Load PDF and create Document objects
    docs = loader.load()

    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100
    )

    # Split documents into chunks
    result = splitter.split_documents(docs)
    print(f"total number of chunk: {len(result)} ")

    # Print chunks
    for i,chunk in enumerate(result):

        print(result[i].metadata)
        print(len(chunk.page_content),"the characters size")
        print("-"* 50 )
        print(chunk.page_content)
        print("-"* 40)






if len(total_text) > 100:
    print("Digital PDF detected")
    print("Use PyMuPDFLoader")





else:
    print("Scanned PDF detected")
    print("Use OCR")



if __name__ == "__main__":
    main(path)