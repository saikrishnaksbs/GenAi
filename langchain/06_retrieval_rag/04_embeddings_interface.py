"""
EMBEDDINGS INTERFACE
====================
An embedding model turns text into a vector of floats such that
semantically similar text ends up close together in vector space. Every
LangChain embeddings integration implements the same `Embeddings` base
class, so you can swap OpenAI for a local HuggingFace model without
touching the rest of your pipeline:

    .embed_query(text)        -> embed a single string (e.g. a user query)
    .embed_documents([texts])  -> embed a batch of strings (e.g. chunks)

Query and document embedding are separate methods because some providers
use asymmetric embeddings (queries and passages are embedded differently
for better retrieval).
"""

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings

# The standard OpenAI embeddings integration; dimension depends on the model.
embeddings: Embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

# Embed a single query string -> one vector.
query_vector = embeddings.embed_query("What is retrieval-augmented generation?")
print(len(query_vector))
# -> 1536
print(query_vector[:3])
# -> [0.0023, -0.0114, 0.0087]

# Embed many documents at once -> list of vectors, one per input string.
# Batching like this is more efficient than calling embed_query in a loop.
chunks = [
    "RAG retrieves relevant context before generating an answer.",
    "Vector stores index embeddings for fast similarity search.",
    "Chunking breaks long documents into retrievable pieces.",
]
doc_vectors = embeddings.embed_documents(chunks)
print(len(doc_vectors), len(doc_vectors[0]))
# -> 3 1536


# You can implement the same interface for a custom or local model.
class FakeDeterministicEmbeddings(Embeddings):
    """Toy embeddings for tests: hashes text into a fixed-size vector."""

    def __init__(self, size: int = 8):
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Deterministic, non-semantic stand-in useful only for wiring tests.
        return [(hash((text, i)) % 1000) / 1000 for i in range(self.size)]


fake_embeddings = FakeDeterministicEmbeddings()
print(len(fake_embeddings.embed_query("hello")))
# -> 8
