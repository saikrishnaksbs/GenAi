"""
Rerankers / Compressors (Cohere Rerank, LLMChainExtractor)
==============================================================
After an initial retrieval pass (which is fast but imprecise), a reranker
re-scores the candidate documents with a more accurate (but slower) model
and keeps only the best ones. `LLMChainExtractor` goes further: it also
trims each document down to just the sentences relevant to the query,
reducing context-window bloat and noise ("contextual compression").
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

vectorstore = FAISS.from_texts(
    [
        "LangChain supports many vector stores like FAISS, Chroma, Pinecone.",
        "Rerankers improve precision after initial vector retrieval.",
        "Paris is the capital of France.",
    ],
    OllamaEmbeddings(model="qwen3-embedding:8b"),
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Option A: Cohere's dedicated reranking model (cross-encoder, high precision) ---
cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=2)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=base_retriever,
)
print(reranking_retriever.invoke("How do rerankers improve RAG?"))

# --- Option B: use an LLM itself to extract only the relevant snippet from each doc ---
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
compressed_docs = compression_retriever.invoke("What vector stores does LangChain support?")
for doc in compressed_docs:
    print(doc.page_content)  # trimmed to just the relevant sentence(s)
