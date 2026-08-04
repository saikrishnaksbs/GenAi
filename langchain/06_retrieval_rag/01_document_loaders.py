"""
DOCUMENT LOADERS
================
Document loaders pull raw content from a source (PDF, CSV, website, Notion
export, database, ...) and convert it into LangChain `Document` objects,
each with `page_content` (the text) and `metadata` (source, page number,
etc). Loaders are the first stage of a RAG pipeline: load -> split ->
embed -> store -> retrieve. Every loader exposes the same two methods:

    .load()       -> returns a list[Document] all at once
    .lazy_load()  -> returns a generator, useful for huge corpora

Swapping a data source is usually just swapping the loader class; the
rest of the pipeline stays the same.
"""

from langchain_community.document_loaders import (
    CSVLoader,
    NotionDirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
)

# --- PDF loader: one Document per page, page number stored in metadata ---
pdf_loader = PyPDFLoader("docs/employee_handbook.pdf")
pdf_docs = pdf_loader.load()

print(len(pdf_docs))
# -> 42
print(pdf_docs[0].metadata)
# -> {"source": "docs/employee_handbook.pdf", "page": 0}

# --- CSV loader: one Document per row, columns joined as key: value text ---
csv_loader = CSVLoader(
    file_path="data/faq.csv",
    source_column="question",  # use this column's value as the metadata "source"
)
csv_docs = csv_loader.load()

print(csv_docs[0].page_content)
# -> "question: How do I reset my password?\nanswer: Go to Settings > Security."

# --- Web loader: fetches a URL and strips it down to visible text ---
web_loader = WebBaseLoader(
    web_paths=["https://example.com/docs/getting-started"],
)
web_docs = web_loader.load()

print(web_docs[0].metadata["title"])
# -> "Getting Started"

# --- Notion-style loader: reads a directory of exported markdown pages ---
notion_loader = NotionDirectoryLoader("data/notion_export")
notion_docs = notion_loader.load()

# lazy_load avoids materializing everything in memory for large exports
for doc in notion_loader.lazy_load():
    # each doc.metadata["source"] points back to the original .md file path
    print(doc.metadata["source"])
    break
# -> "data/notion_export/Engineering Wiki.md"

# All loaders funnel into the same downstream shape: list[Document]
all_docs = pdf_docs + csv_docs + web_docs + notion_docs
print(f"Loaded {len(all_docs)} documents from 4 sources")
# -> "Loaded 45 documents from 4 sources"
