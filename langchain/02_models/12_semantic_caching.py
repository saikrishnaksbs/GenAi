"""
SEMANTIC CACHING
================
Standard exact-match caching (like SQLiteCache) fails if a user modifies their query
even slightly (e.g. "What is the capital of France?" vs "Tell me the capital of France").
Semantic caching fixes this by storing prompt embeddings and querying a vector store.
If a new query is semantically similar to a cached query (similarity >= threshold),
the cache returns the stored response.

This script implements a custom `LocalSemanticCache` subclassing LangChain's `BaseCache`
to demonstrate the inner mechanics of semantic caching without requiring external services like Redis.
"""

from typing import Optional, Sequence
import numpy as np
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation
from langchain_core.globals import set_llm_cache
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Helper to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class LocalSemanticCache(BaseCache):
    """A simple in-memory semantic cache using LangChain Embeddings."""
    
    def __init__(self, embeddings_model, threshold: float = 0.85):
        self.embeddings = embeddings_model
        self.threshold = threshold
        # Store items as: {"prompt": str, "embedding": list[float], "generations": list[Generation]}
        self.cache_store = []

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        """Look up a prompt semantically."""
        if not self.cache_store:
            return None
            
        # Get embedding of the incoming prompt
        query_vector = self.embeddings.embed_query(prompt)
        
        best_similarity = -1.0
        best_generations = None
        best_prompt = ""
        
        # Search the cache for the most similar prompt
        for item in self.cache_store:
            similarity = cosine_similarity(query_vector, item["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_generations = item["generations"]
                best_prompt = item["prompt"]
                
        if best_similarity >= self.threshold:
            print(f"\n[Semantic Cache HIT] Similarity: {best_similarity:.4f}")
            print(f"Matched prompt: '{best_prompt}' for query: '{prompt}'")
            return best_generations
            
        print(f"\n[Semantic Cache MISS] Best similarity: {max(best_similarity, 0.0):.4f} for query: '{prompt}'")
        return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        """Update the cache store with a new prompt and its generations."""
        # Check if already exists exactly to prevent duplicate entries
        for item in self.cache_store:
            if item["prompt"] == prompt:
                item["generations"] = return_val
                return
                
        embedding = self.embeddings.embed_query(prompt)
        self.cache_store.append({
            "prompt": prompt,
            "embedding": embedding,
            "generations": return_val
        })
        print(f"[Semantic Cache POPULATED] Cached: '{prompt}'")

    def clear(self) -> None:
        """Clear the cache store."""
        self.cache_store.clear()


# --------------------------------------------------------------------------
# Demonstration of Custom Semantic Cache
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize OpenAI Embeddings and Model
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

    # Instantiate and install our custom cache
    semantic_cache = LocalSemanticCache(embeddings_model=embeddings, threshold=0.90)
    set_llm_cache(semantic_cache)

    # 1. First invocation: Cache Miss
    q1 = "What is the distance between the Earth and the Moon?"
    print(f"\nPrompt: {q1}")
    res1 = model.invoke(q1)
    print(f"Response: {res1.content}")

    # 2. Second invocation: Different phrasing (Semantic Hit)
    q2 = "How far is the Moon from the Earth?"
    print(f"\nPrompt: {q2}")
    res2 = model.invoke(q2)
    print(f"Response (Cached): {res2.content}")

    # 3. Third invocation: Unrelated query (Cache Miss)
    q3 = "What is the capital of Japan?"
    print(f"\nPrompt: {q3}")
    res3 = model.invoke(q3)
    print(f"Response: {res3.content}")
