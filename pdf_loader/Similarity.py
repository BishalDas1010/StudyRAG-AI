try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    HAS_COMPRESSION = True
except Exception:
    ContextualCompressionRetriever = None
    LLMChainExtractor = None
    HAS_COMPRESSION = False


class Similarity_search:

    def __init__(
        self,
        vector_db,
        query="",
        RETRIEVER_K=5,
        MMR_LAMBDA=0.5,
    ):
        self.vector_db = vector_db
        self.query = query
        self.RETRIEVER_K = RETRIEVER_K
        self.MMR_LAMBDA = MMR_LAMBDA

    def contextualCompressionRetriever(self, llm, query=None):
        search_query = query or self.query

        base_retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.RETRIEVER_K,
                "fetch_k": 20,
                "lambda_mult": self.MMR_LAMBDA,
            },
        )

        if HAS_COMPRESSION and ContextualCompressionRetriever is not None and LLMChainExtractor is not None:
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            return compression_retriever.invoke(search_query)

        return base_retriever.invoke(search_query)