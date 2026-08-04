"""
DOCUMENT OBJECT
===============
`Document` is the common currency of every retrieval component in
LangChain: loaders produce them, splitters chop them up, vector stores
embed and store them, and retrievers hand them back to your chain. It is
a deliberately simple container:

    page_content: str        -> the actual text
    metadata: dict            -> arbitrary structured info about the text
    id: Optional[str]         -> optional stable identifier

Keeping metadata alongside content is what lets you filter searches,
cite sources, and trace an answer back to where it came from.
"""

from langchain_core.documents import Document

# The minimal case: just text.
doc = Document(page_content="LangChain makes it easy to compose LLM pipelines.")
print(doc.page_content)
# -> "LangChain makes it easy to compose LLM pipelines."

# Metadata is what makes retrieval useful in practice: source attribution,
# filtering ("only search docs from 2024"), and citation links all rely on it.
doc_with_metadata = Document(
    page_content="Q3 revenue grew 12% year over year.",
    metadata={
        "source": "reports/q3_2025.pdf",
        "page": 4,
        "author": "finance-team",
        "year": 2025,
    },
    id="doc-001",  # stable id, useful for updates/deduplication in a vector store
)
print(doc_with_metadata.metadata["source"])
# -> "reports/q3_2025.pdf"
print(doc_with_metadata.id)
# -> "doc-001"

# Documents are plain data objects, so you can freely transform them.
def add_ingestion_timestamp(document: Document, timestamp: str) -> Document:
    # Return a new Document rather than mutating in place, to keep transforms pure.
    return Document(
        page_content=document.page_content,
        metadata={**document.metadata, "ingested_at": timestamp},
        id=document.id,
    )


stamped = add_ingestion_timestamp(doc_with_metadata, "2026-08-01T00:00:00Z")
print(stamped.metadata)
# -> {"source": "reports/q3_2025.pdf", "page": 4, "author": "finance-team",
#     "year": 2025, "ingested_at": "2026-08-01T00:00:00Z"}

# A list[Document] is what flows between every retrieval-pipeline stage.
corpus = [doc, doc_with_metadata, stamped]
sources = [d.metadata.get("source", "unknown") for d in corpus]
print(sources)
# -> ["unknown", "reports/q3_2025.pdf", "reports/q3_2025.pdf"]
