"""
VECTORSTORE INTERFACE
=====================
A vector store indexes embedded documents so you can find the ones most
similar to a query vector. LangChain wraps many backends (Chroma,
FAISS, Pinecone, PGVector, ...) behind one common interface, so you can
develop locally with Chroma and swap in a hosted store for production
with minimal code changes:

    .add_documents(docs)                    -> index new documents
    .similarity_search(query, k)             -> top-k most similar Documents
    .similarity_search_with_score(query, k)  -> same, plus distance/score
    .max_marginal_relevance_search(query, k) -> diverse top-k results (MMR)

Under the hood, the store embeds your query with the same embeddings
model used to index the documents, then does a nearest-neighbor search.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

docs = [
    Document(page_content="Chroma is an open-source embedding database.", metadata={"topic": "storage"}),
    Document(page_content="FAISS is a library for efficient similarity search.", metadata={"topic": "storage"}),
    Document(page_content="LangChain chains compose runnables with the | operator.", metadata={"topic": "core"}),
]

# Creating a store from documents embeds and indexes them in one call.
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="demo_collection",
    persist_directory="./chroma_db",  # omit for an in-memory, ephemeral store
)

# You can also add documents to an existing store incrementally.
new_docs = [Document(page_content="Pinecone is a managed vector database.", metadata={"topic": "storage"})]
vectorstore.add_documents(new_docs)

# --- Plain similarity search: top-k nearest neighbors by embedding distance ---
results = vectorstore.similarity_search("What vector databases are available?", k=2)
for r in results:
    print(r.page_content)
# -> "Chroma is an open-source embedding database."
# -> "Pinecone is a managed vector database."

# --- With scores: useful for filtering out weak matches below a threshold ---
scored_results = vectorstore.similarity_search_with_score("vector database", k=2)
for doc, score in scored_results:
    # Chroma returns a distance (lower = more similar) for its default metric.
    print(doc.page_content, score)
# -> "Chroma is an open-source embedding database." 0.21
# -> "Pinecone is a managed vector database." 0.24

# --- Metadata filtering: restrict the search to a subset of documents ---
filtered = vectorstore.similarity_search("database", k=5, filter={"topic": "storage"})
print(len(filtered))
# -> 3

# --- MMR: Maximal Marginal Relevance trades some relevance for diversity ---
# Useful when top results are near-duplicates and you want varied context.
mmr_results = vectorstore.max_marginal_relevance_search(
    "vector database",
    k=2,
    fetch_k=10,     # pull this many candidates before re-ranking for diversity
    lambda_mult=0.5,  # 0 = max diversity, 1 = max relevance
)
print([d.page_content for d in mmr_results])
# -> ["Chroma is an open-source embedding database.", "FAISS is a library for efficient similarity search."]
