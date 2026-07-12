from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



class Pipeline:

    def __init__(self, pdf):
        self.path = pdf


    def detect_pdf_type(self):
        loader = PyMuPDFLoader(self.path)
        documents = loader.load()

        total_text = "".join(
            doc.page_content.strip()
            for doc in documents
        )

        if len(total_text) > 100:
            print("Digital PDF detected")
            return True

        else:
            print("Scanned PDF detected")
            print("Use OCR")
            return False


    def pdf_loader(self):
        loader = PyMuPDFLoader(self.path)
        docs = loader.load()
        if not docs:
            print("SOme thing worng here so the docss is empty")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        print(f"Total number of chunks: {len(chunks)}")

        return chunks


    def embedding(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cuda"
            }
        )

        return embedding_model
    

    def chroma_db(self, embedding_model, docs):

    # Prevent empty document insertion
        if not docs:
            print(" No documents found. Skipping ChromaDB insertion.")
            return None

        print(f"Adding {len(docs)} documents to ChromaDB")

        vector_store = Chroma(
            collection_name="pdf_collection",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )

        vector_store.add_documents(docs)

        print(" Documents successfully added to ChromaDB")

        return vector_store


    # def chroma_db(self, embedding_model, docs):
    #     vector_store = Chroma(
    #         collection_name="study_rag",
    #         embedding_function=embedding_model,
    #         persist_directory="./chroma_study_db"
    #     )

    

        # vector_store.add_documents(docs)

        # print("Documents added and stored successfully")

        # return vector_store
