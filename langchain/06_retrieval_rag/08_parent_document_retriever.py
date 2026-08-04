"""
PARENT DOCUMENT RETRIEVER
==========================
Small chunks embed and match queries well, but large chunks give the LLM
more surrounding context to reason with. `ParentDocumentRetriever`
resolves this tension: it indexes small "child" chunks for accurate
similarity search, but returns their larger "parent" chunk (or the
whole original document) to the LLM. Children live in a vector store;
parents live in a separate, simple key-value docstore.
"""

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Splitter for the big chunks that get returned to the LLM as context.
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

# Splitter for the small chunks that actually get embedded and searched.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectorstore = Chroma(collection_name="child_chunks", embedding_function=embeddings)

# The docstore holds full parent documents, keyed by an auto-generated id.
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,  # omit this to use whole source docs as parents
)

source_docs = [
    Document(
        page_content=(
            "LangChain Expression Language (LCEL) is a declarative way to "
            "compose chains. " * 50
        ),
        metadata={"source": "lcel_guide.txt"},
    )
]

# add_documents splits into parents, splits each parent into children,
# embeds only the children, and stores parent<->child linkage internally.
retriever.add_documents(source_docs, ids=None)

# Search happens against the small, precise child embeddings...
results = retriever.invoke("What is LCEL?")

# ...but what you get back is the larger parent chunk, giving the LLM
# more surrounding context than the tiny matched snippet alone.
print(len(results))
# -> 1
print(len(results[0].page_content))
# -> 2000

# You can inspect the raw child chunks that were actually embedded/matched
# via the underlying vector store, useful for debugging retrieval quality.
child_matches = vectorstore.similarity_search("What is LCEL?", k=1)
print(len(child_matches[0].page_content))
# -> 400
