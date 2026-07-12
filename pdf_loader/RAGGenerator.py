from langchain_core.prompts import ChatPromptTemplate

class RAGGenerator:

    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer using only the given context.

Context:
{context}

Question:
{question}

Answer:
""")

    def generate(self, docs, question):

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        chain = self.prompt | self.llm

        return chain.invoke({
            "context": context,
            "question": question
        })