"""
MULTI QUERY RETRIEVER
======================
A single user query might not match a document's wording even when the
document is relevant ("vocabulary mismatch"). `MultiQueryRetriever` asks
an LLM to rewrite the original query into several different phrasings,
runs retrieval for each variant against the underlying retriever, and
returns the union of unique results. This broadens recall at the cost
of extra LLM calls and retrieval requests.
"""

import logging

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectorstore = Chroma(collection_name="demo_collection", embedding_function=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)  # low temperature for consistent rewrites

# Wraps the base retriever; the LLM generates alternate phrasings of the query.
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# Enable logging to see the generated query variants — helpful for debugging
# why certain documents are (or aren't) being retrieved.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

results = multi_query_retriever.invoke("What are the benefits of RAG?")
# Internally the LLM might generate variants like:
#   "Why use retrieval augmented generation?"
#   "What advantages does RAG provide over plain LLM prompting?"
#   "How does RAG improve answer accuracy?"
# ...then queries the vector store with each and de-duplicates the results.

print(len(results))
# -> 5
for doc in results:
    print(doc.page_content[:60])
# -> "RAG grounds LLM answers in retrieved, up-to-date documents"
# -> "Retrieval reduces hallucination by supplying real context"
# -> ...

# You can also supply a custom prompt to control how variants are generated,
# e.g. to bias toward domain-specific synonyms.
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template(
    "Generate 3 different search queries related to: {question}\n"
    "Provide one query per line, no numbering."
)

custom_multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=custom_prompt,
)
