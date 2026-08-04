"""
Indexing API (index() for incremental sync and dedup)
=========================================================
Re-embedding your entire document set every time a few source documents
change is wasteful and expensive. LangChain's `index()` API syncs a list of
documents into a vector store while tracking what's already indexed (via a
`RecordManager`), so it can:
  - skip documents that are unchanged (by content hash)
  - add only new/changed documents
  - delete vectors for documents that were removed from the source
"""

from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

collection_name = "my_docs"
vectorstore = Chroma(collection_name=collection_name, embedding_function=OllamaEmbeddings(model="qwen3-embedding:8b"))

# RecordManager tracks which document hashes have already been indexed,
# so re-running index() on unchanged docs is a cheap no-op.
record_manager = SQLRecordManager(
    namespace=f"chroma/{collection_name}",
    db_url="sqlite:///record_manager_cache.sql",
)
record_manager.create_schema()

docs = [
    Document(page_content="LangChain indexing avoids duplicate embeddings.", metadata={"source": "doc1.txt"}),
    Document(page_content="Incremental sync only re-embeds changed content.", metadata={"source": "doc2.txt"}),
]

# cleanup="incremental": deletes vectors whose source docs disappeared from `docs`,
# but only within sources seen in this batch. Use cleanup="full" to sync against
# the ENTIRE source set (deletes anything not present at all).
result = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print(result)
# -> {"num_added": 2, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

# Running index() again with the SAME docs is nearly free — everything is skipped.
result_second_run = index(docs, record_manager, vectorstore, cleanup="incremental", source_id_key="source")
print(result_second_run)
# -> {"num_added": 0, "num_updated": 0, "num_skipped": 2, "num_deleted": 0}
