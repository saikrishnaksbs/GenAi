"""
EnsembleRetriever (Hybrid Search)
====================================
Combines multiple retrievers — typically a sparse/keyword retriever (BM25)
and a dense/semantic retriever (vector similarity) — and merges their
results using Reciprocal Rank Fusion. This "hybrid search" usually beats
either method alone: BM25 catches exact keyword/ID matches that embeddings
miss, while vector search catches semantic paraphrases that BM25 misses.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever

docs = [
    Document(page_content="LangChain provides retrievers for RAG pipelines."),
    Document(page_content="BM25 is a classic sparse keyword-based ranking algorithm."),
    Document(page_content="Vector embeddings capture semantic similarity between texts."),
]

# Sparse retriever: exact keyword matching, great for names/IDs/jargon.
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

# Dense retriever: semantic similarity via embeddings.
vectorstore = FAISS.from_documents(docs, OllamaEmbeddings(model="qwen3-embedding:8b"))
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# weights control how much each retriever's ranking contributes to the fused score.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("keyword search algorithms")
for doc in results:
    print(doc.page_content)
