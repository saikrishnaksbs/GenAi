"""
TEXT SPLITTERS
==============
LLMs and embedding models have limited context windows, so long documents
need to be broken into smaller chunks before embedding. Text splitters
handle this "chunking" step. Choosing chunk size and overlap affects
retrieval quality: too big and irrelevant text dilutes the match, too
small and you lose context. Different splitters respect different
document structures (plain text, tokens, markdown headings, HTML tags).
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

long_text = """LangChain is a framework for building LLM applications.
It provides abstractions for prompts, models, memory, and retrieval.
RAG combines retrieval with generation to ground answers in real data.
""" * 20  # simulate a long document

# --- RecursiveCharacterTextSplitter: the default, general-purpose choice ---
# Tries to split on paragraph, then sentence, then word boundaries in order,
# only falling back to a hard character cut if nothing else fits.
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # max characters per chunk
    chunk_overlap=20,     # characters shared between consecutive chunks for continuity
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = recursive_splitter.split_text(long_text)
print(len(chunks))
# -> 9
print(chunks[0][:50])
# -> "LangChain is a framework for building LLM applic"

# split_documents preserves and copies metadata onto each resulting chunk
docs = [Document(page_content=long_text, metadata={"source": "intro.txt"})]
split_docs = recursive_splitter.split_documents(docs)
print(split_docs[0].metadata)
# -> {"source": "intro.txt"}

# --- TokenTextSplitter: splits by model tokens, not characters ---
# Useful when you need chunks to fit precisely under an embedding model's
# token limit rather than an approximate character count.
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # tokenizer used by OpenAI embedding models
    chunk_size=50,
    chunk_overlap=5,
)
token_chunks = token_splitter.split_text(long_text)
print(len(token_chunks))
# -> 6

# --- MarkdownHeaderTextSplitter: splits on heading structure, keeps hierarchy ---
markdown_text = "# Guide\n## Setup\nInstall the package.\n## Usage\nCall .invoke()."
headers_to_split_on = [("#", "h1"), ("##", "h2")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(markdown_text)
print(md_chunks[1].metadata)
# -> {"h1": "Guide", "h2": "Usage"}

# --- HTMLHeaderTextSplitter: same idea, but for HTML tags like <h1>/<h2> ---
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "h1"), ("h2", "h2")])
html_chunks = html_splitter.split_text(
    "<h1>Guide</h1><h2>Setup</h2><p>Install the package.</p>"
)
print(html_chunks[0].page_content)
# -> "Install the package."
