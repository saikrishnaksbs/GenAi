"""
RETRIEVER INTERFACE
====================
A `Retriever` is anything that takes a text query and returns a list of
relevant `Document` objects. Like every other LangChain component, it
implements the `Runnable` protocol, so it plugs directly into LCEL
chains with `.invoke()`:

    .invoke(query)   -> list[Document]

A vector store isn't itself a retriever, but `.as_retriever()` wraps it
into one. You can also subclass `BaseRetriever` to wrap any custom
lookup logic (a keyword search, a SQL query, an API call) in the same
uniform interface.
"""

from typing import List

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectorstore = Chroma(collection_name="demo_collection", embedding_function=embeddings)

# --- The common case: turn a vector store into a retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",     # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 3},
)

# Retrievers are Runnables, so .invoke() is the standard call, just like
# a chat model or a chain.
results = retriever.invoke("How does LangChain handle retrieval?")
print(len(results))
# -> 3
print(results[0].page_content)
# -> "LangChain retrievers implement a common .invoke() interface."

# --- A custom retriever: wrap arbitrary lookup logic behind the same API ---
class KeywordRetriever(BaseRetriever):
    """Toy retriever that returns documents containing all query words."""

    documents: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_words = query.lower().split()
        return [
            doc
            for doc in self.documents
            if all(word in doc.page_content.lower() for word in query_words)
        ]


keyword_retriever = KeywordRetriever(
    documents=[
        Document(page_content="LangChain supports retrieval augmented generation."),
        Document(page_content="Chains compose prompts, models, and parsers."),
    ]
)

matches = keyword_retriever.invoke("retrieval generation")
print(len(matches))
# -> 1

# Because both are Runnables, they can be swapped in a chain without any
# other code changing — this is the whole point of the common interface.
def use_any_retriever(r: BaseRetriever, query: str) -> int:
    return len(r.invoke(query))


print(use_any_retriever(retriever, "vector search"))
# -> 3
