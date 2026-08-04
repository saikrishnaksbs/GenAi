"""
BUILDING A CUSTOM RETRIEVER FROM SCRATCH
========================================
In real-world applications, you often need to fetch background documents or facts
from proprietary sources (e.g. relational SQL databases, custom Elasticsearch clusters,
or internal Web REST APIs) rather than simple vector databases.

To connect external database resources, you subclass LangChain's **`BaseRetriever`**
and override the `_get_relevant_documents` hook.
"""

from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Mock internal database dictionary
MOCK_KNOWLEDGE_BASE = [
    {"title": "Refund Policy", "text": "Customers can request a refund within 30 days of purchase with receipt."},
    {"title": "Shipping Times", "text": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days."},
    {"title": "International Taxes", "text": "International shipments are subject to local customs taxes and tariffs."}
]


class CustomDatabaseRetriever(BaseRetriever):
    """A custom retriever that queries a proprietary database using keyword checks."""
    
    database: List[dict]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronously retrieves relevant documents by matching terms inside the query."""
        results = []
        query_lower = query.lower()
        
        for record in self.database:
            # Check if any database keywords overlap with the query string
            # In a real environment, you would write: cursor.execute("SELECT ... WHERE ...")
            if any(term in query_lower for term in record["title"].lower().split()):
                doc = Document(
                    page_content=record["text"],
                    metadata={"source": "custom_db", "category": record["title"]}
                )
                results.append(doc)
                
        return results

    # Optionally override _aget_relevant_documents for async speedups
    # async def _aget_relevant_documents(...)


# --------------------------------------------------------------------------
# Demonstration of Custom Retriever inside RAG LCEL Chain
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize Custom Retriever
    my_retriever = CustomDatabaseRetriever(database=MOCK_KNOWLEDGE_BASE)
    
    # Run standalone retriever test
    print("--- Standalone Retriever Query Output ---")
    retrieved_docs = my_retriever.invoke("What is the refund process?")
    for doc in retrieved_docs:
        print(f"Doc: {doc.metadata['category']} -> '{doc.page_content}'")
        
    # 2. Integrate into standard LCEL RAG pipeline
    model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the user question using ONLY the provided context details. If you do not know, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    # Helper to combine retrieved documents to a single string
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {
            # Retriever executes and the result gets piped to format_docs
            "context": my_retriever | format_docs,
            "question": lambda x: x["question"]
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    print("\n--- RAG Chain Invocation Response ---")
    res = rag_chain.invoke({"question": "Are shipping times fast?"})
    print(res)
