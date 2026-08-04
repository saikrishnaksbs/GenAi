"""
BATCH INGESTION STRATEGIES FOR LARGE DOCUMENT SETS
==================================================
When importing thousands or millions of documents into a production RAG system,
ingesting them one-by-one synchronously results in high latency, network timeouts,
and rate-limiting errors from embedding providers.

This script demonstrates production batch ingestion strategies:
1. Slicing documents into fixed-size batches.
2. Concurrent batch ingestion using Python's `ThreadPoolExecutor`.
3. High-throughput async batch upload via vectorstore `.aadd_documents()`.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Generate mock documents (e.g., 250 documents)
large_doc_set = [
    Document(
        page_content=f"This is document number {i}. It contains information about transaction logs.",
        metadata={"doc_index": i, "category": "logs"}
    )
    for i in range(250)
]


# Helper to slice lists into smaller batches
def chunk_list(lst: list, size: int) -> List[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# --------------------------------------------------------------------------
# Strategy 1: Threaded Concurrent Batch Ingestion
# --------------------------------------------------------------------------
def ingest_batch_sync(vectorstore: FAISS, batch: List[Document]) -> None:
    """Ingests a single batch synchronously (designed to run inside threads)."""
    # FAISS.add_documents handles embedding calculation and storage
    vectorstore.add_documents(batch)
    print(f"[Thread Ingested] Batch of {len(batch)} documents processed.")


def run_threaded_ingestion(docs: List[Document], batch_size: int = 50, max_workers: int = 4):
    """Orchestrates ingestion across a thread pool."""
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    # Initialize an empty FAISS vectorstore
    db = FAISS.from_documents([Document(page_content="initialization")], embeddings)
    
    batches = chunk_list(docs, batch_size)
    print(f"\n--- Strategy 1: Threaded Ingestion (Batches: {len(batches)}, Workers: {max_workers}) ---")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in batches:
            executor.submit(ingest_batch_sync, db, batch)
            
    print(f"Ingestion completed in {time.time() - start_time:.2f} seconds.")
    return db


# --------------------------------------------------------------------------
# Strategy 2: High-Performance Async Ingestion
# --------------------------------------------------------------------------
async def run_async_ingestion(docs: List[Document], batch_size: int = 50):
    """Performs non-blocking async ingestion using asyncio tasks."""
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    db = FAISS.from_documents([Document(page_content="initialization")], embeddings)
    
    batches = chunk_list(docs, batch_size)
    print(f"\n--- Strategy 2: Async Ingestion (Batches: {len(batches)}) ---")
    
    start_time = time.time()
    
    # Create coroutines for async uploads
    async def upload_task(batch_docs):
        # Utilizes non-blocking async network I/O
        await db.aadd_documents(batch_docs)
        print(f"[Async Ingested] Batch of {len(batch_docs)} documents.")

    # Execute all tasks concurrently
    tasks = [upload_task(b) for b in batches]
    await asyncio.gather(*tasks)
    
    print(f"Async Ingestion completed in {time.time() - start_time:.2f} seconds.")
    return db


if __name__ == "__main__":
    # Test Threaded Ingestion
    run_threaded_ingestion(large_doc_set, batch_size=50)
    
    # Test Async Ingestion
    asyncio.run(run_async_ingestion(large_doc_set, batch_size=50))
