# Retrieval-Augmented Generation (RAG) & Retrieval

This directory details LangChain's **Retrieval-Augmented Generation (RAG)** stack. It covers the full lifecycle of data integration: loading documents, chunking text, generating embeddings, vector storage, retriever abstractions, advanced retrieval strategies, incremental indexing/syncing, and fully compiled QA chains.

---

## Table of Contents
1. [RAG Lifecycle & The Document Object](#1-rag-lifecycle--the-document-object)
2. [Document Loaders & Text Splitters](#2-document-loaders--text-splitters)
3. [Embeddings & Vectorstore Interfaces](#3-embeddings--vectorstore-interfaces)
4. [The Retriever Interface](#4-the-retriever-interface)
5. [Advanced Retrieval Strategies](#5-advanced-retrieval-strategies)
6. [Rerankers & Compressors](#6-rerankers--compressors)
7. [The Indexing API (Incremental Sync)](#7-the-indexing-api-incremental-sync)
8. [Putting It Together: Full RAG Chain](#8-putting-it-together-full-rag-chain)
9. [Building Custom Retrievers](#9-building-custom-retrievers)
10. [Batch Ingestion at Scale](#10-batch-ingestion-at-scale)
11. [Multi-Modal RAG (Images & Tables)](#11-multi-modal-rag-images--tables)

---

## 1. RAG Lifecycle & The Document Object
Retrieval is built around the [Document](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/03_document_object.py#L3) object, which acts as the common currency between loaders, splitters, vector stores, and retrievers. A `Document` holds the following fields:
- `page_content`: The raw text string.
- `metadata`: A dictionary holding structured attributes (e.g., `source`, `page_number`).
- `id`: An optional stable unique identifier.

---

## 2. Document Loaders & Text Splitters

### Document Loaders
[Document Loaders](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/01_document_loaders.py#L4-L8) fetch raw files (PDFs, CSVs, Webpages) and convert them to `Document` lists.
- `load()`: Returns all documents in memory at once.
- `lazy_load()`: Returns a generator, loading documents on-demand to process large datasets without exceeding RAM constraints.

### Text Splitters
[Text Splitters](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/02_text_splitters.py#L4-L8) break large text into smaller chunks.
- Tuning **`chunk_size`** and **`chunk_overlap`** prevents data loss and ensures key context is captured without introducing excess noise into semantic similarity scores.
- Different splitters exist for plain text, Markdown headers, HTML nodes, and specific tokens.

---

## 3. Embeddings & Vectorstore Interfaces

### Embeddings Interface
[Embeddings](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/04_embeddings_interface.py#L4-L10) convert text into a numeric vector representing semantic intent. The interface separates query and document embedding:
- `embed_query(text)`: For query processing.
- `embed_documents(list[text])`: For bulk document indexing.
This separation allows models to run asymmetric embeddings where queries and passage texts are embedded differently.

### Vectorstore Interface
[Vector Stores](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/05_vectorstore_interface.py#L4-L10) store document vectors and perform nearest-neighbor queries:
- `add_documents(docs)`: Indexes new documents.
- `similarity_search(query, k)`: Returns the top `k` most similar documents.
- `similarity_search_with_score(query, k)`: Returns documents and their similarity scores.
- `max_marginal_relevance_search(query, k)` (MMR): Optimizes for both query similarity and document diversity, reducing redundancy in the retrieved context.

---

## 4. The Retriever Interface
A [Retriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/06_retriever_interface.py#L4-L10) is a Runnable that accepts a text query and returns a list of `Document` objects.
- Vector stores can be wrapped as a retriever with `.as_retriever(search_kwargs={"k": 2})`.
- Subclassing `BaseRetriever` lets you implement custom search APIs or databases.

---

## 5. Advanced Retrieval Strategies
To bypass the limitations of basic similarity searches, LangChain supports advanced retrieval components:
- **[MultiQueryRetriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/07_multi_query_retriever.py#L4-L10)**: Uses an LLM to rewrite the user's query into multiple versions. It retrieves candidates for all variants and returns the union of unique results to prevent vocabulary mismatch issues.
- **[ParentDocumentRetriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/08_parent_document_retriever.py#L4-L10)**: Splits documents into small "child" chunks for accurate embedding matches, but returns the larger "parent" document context to the LLM.
- **[SelfQueryRetriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/09_self_query_retriever.py#L4-L10)**: Uses an LLM to extract a query into a semantic string and a structured metadata filter, applying the filter directly at the vector store level.
- **[ContextualCompressionRetriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/10_contextual_compression_retriever.py#L4-L10)**: Wraps a base retriever and runs a document compressor that filters or trims irrelevant sentences before returning the documents.
- **[EnsembleRetriever](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/11_ensemble_retriever.py#L4-L9)**: Runs hybrid searches (e.g. BM25 keyword matching + dense semantic vector search) and fuses results using Reciprocal Rank Fusion (RRF).

---

## 6. Rerankers & Compressors
After initial candidate retrieval (which prioritizes speed), you can apply a [Rerank pass](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/12_rerankers_and_compressors.py#L4-L8) using a cross-encoder model (e.g. Cohere Rerank) or an LLM extractor (like `LLMChainExtractor`). This scoring step filters and trims text chunks to keep only the highest-quality segments.

---

## 7. The Indexing API (Incremental Sync)
Re-indexing an entire database is slow and expensive. The [Indexing API](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/13_indexing_api.py#L4-L10) manages incremental syncs:
- Tracks document state using a **`RecordManager`**.
- Skips documents whose content hashes have not changed.
- Updates edited items and automatically deletes vectors of documents that have been removed from the source data.

---

## 8. Putting It Together: Full RAG Chain
The modern way to implement RAG chains combines two helper functions (see [14_full_rag_chain.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/14_full_rag_chain.py)):
- **`create_stuff_documents_chain(model, prompt)`**: Takes retrieved documents and injects their content directly into the `{context}` slot of the prompt.
- **`create_retrieval_chain(retriever, document_chain)`**: Wires the retriever to the document chain, returns the final text response under `"answer"`, and provides the raw source documents list under `"context"` for citations.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

document_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Invocation returns a dict containing both keys
response = rag_chain.invoke({"input": "What is LangGraph built on?"})
print(response["answer"])
print(response["context"]) # list[Document]
```

---

## 9. Building Custom Retrievers
When background knowledge is stored in proprietary relational databases or REST APIs instead of standard vector stores, you must write a custom retriever.
- Subclass **`BaseRetriever`** from `langchain_core.retrievers`.
- Override the synchronous **`_get_relevant_documents`** (or asynchronous `_aget_relevant_documents`) to execute your query logic and return a list of `Document` objects.

See [15_custom_retriever.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/15_custom_retriever.py) for examples.

---

## 10. Batch Ingestion at Scale
Ingesting millions of documents synchronously blocks execution threads and triggers rate limits.
- **Fixed-Size Chunking**: Slice datasets into uniform batches (e.g. 50-100 items).
- **Concurrency**: Parallelize operations across threads (`ThreadPoolExecutor`) or async network I/O requests (`aadd_documents`).

See [16_batch_ingestion.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/16_batch_ingestion.py) for details.

---

## 11. Multi-Modal RAG (Images & Tables)
Extracting knowledge from diagrams, layout elements, and inline tables.
- **Multimodal Descriptions**: Generate summaries describing visual charts using a multimodal model (like `gpt-4o`).
- **Hybrid Storage**: Index text documents alongside image descriptions in a single Vector Store, maintaining reference metadata paths back to original file assets.
- **Image Content Contexts**: Inject retrieved summaries and target image references directly into the final LLM prompt context.

See [17_multimodal_rag.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/06_retrieval_rag/17_multimodal_rag.py) for details.
