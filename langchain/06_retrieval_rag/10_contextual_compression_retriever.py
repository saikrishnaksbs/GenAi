"""
CONTEXTUAL COMPRESSION RETRIEVER
=================================
A plain similarity search often returns whole chunks where only a
sentence or two is actually relevant to the query. Feeding the full,
noisy chunks to the LLM wastes context tokens and can dilute the answer.
`ContextualCompressionRetriever` wraps a base retriever with a
"document compressor" that filters or trims each retrieved document
down to just the relevant parts before returning them.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectorstore = Chroma(collection_name="demo_collection", embedding_function=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- LLMChainExtractor: asks the LLM to extract only the relevant sentences ---
# from each document, discarding the rest of the chunk's text.
extractor_compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor_compressor,
    base_retriever=base_retriever,
)

results = compression_retriever.invoke("What year was LangChain founded?")
for doc in results:
    # Each returned Document is now a trimmed excerpt, not the full original chunk.
    print(doc.page_content)
# -> "LangChain was founded in 2022."

# --- LLMChainFilter: cheaper alternative that keeps or drops whole documents ---
# (yes/no relevance check) instead of extracting excerpts — fewer tokens,
# less precise trimming.
filter_compressor = LLMChainFilter.from_llm(llm)

filtering_retriever = ContextualCompressionRetriever(
    base_compressor=filter_compressor,
    base_retriever=base_retriever,
)

filtered_results = filtering_retriever.invoke("What year was LangChain founded?")
print(len(filtered_results))
# -> 2 (irrelevant chunks dropped entirely, relevant ones kept unmodified)

# Compressors can also be chained together, e.g. extract relevant text
# and then deduplicate near-identical results, using DocumentCompressorPipeline.
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, extractor_compressor]
)

pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)
