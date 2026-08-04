"""
Summarization Chains: stuff, map-reduce, refine
===================================================
Three classic strategies for summarizing documents that may not fit in a
single context window:

- "stuff": concatenate all docs into one prompt. Simple, fast, but breaks
  once total content exceeds the context window.
- "map_reduce": summarize each doc chunk independently (map), then summarize
  the summaries (reduce). Scales to large doc sets, parallelizable.
- "refine": summarize the first chunk, then iteratively refine that summary
  by feeding it the next chunk. Preserves more nuance/order but is sequential
  (cannot parallelize) and slower.
"""

from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

long_text = "LangChain is a framework for LLM apps... " * 200  # pretend long document

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(long_text)]

model = ChatOllama(model="qwen2.5:1.5b")

# --- stuff: only viable for small doc sets ---
stuff_chain = load_summarize_chain(model, chain_type="stuff")
# print(stuff_chain.invoke({"input_documents": docs[:2]})["output_text"])

# --- map_reduce: summarize each chunk, then summarize the summaries ---
map_reduce_chain = load_summarize_chain(model, chain_type="map_reduce")
result = map_reduce_chain.invoke({"input_documents": docs})
print(result["output_text"])

# --- refine: sequentially build up one running summary ---
refine_chain = load_summarize_chain(model, chain_type="refine")
refine_result = refine_chain.invoke({"input_documents": docs})
print(refine_result["output_text"])
