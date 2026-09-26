"""
SUMMARIZATION STRATEGIES (Modern LCEL Replacement for load_summarize_chain)
=============================================================================
Legacy LangChain used `load_summarize_chain` to perform Stuff and Map-Reduce summarization.
In modern LangChain, we construct these workflows explicitly using LCEL. This provides
full visibility and control over prompts, concurrency, and intermediate steps.

1. "Stuff" strategy: Concatenates all documents and sends them in a single prompt.
2. "Map-Reduce" strategy: Summarizes each document individually in parallel (Map),
   then combines those summaries into a final overview (Reduce).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# Setup model and sample long text
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

long_text = (
    "LangChain is a framework designed to simplify the creation of applications using large language models. "
    "It provides integrations with various LLM providers, vector databases, and tools. "
    "By establishing standard interfaces for chains, prompts, and memory, LangChain allows developers to quickly build agents, chatbots, and RAG pipelines. "
    "As the LLM ecosystem has matured, LangChain has transitioned to LangChain Expression Language (LCEL) as the core composition paradigm, "
    "enabling automatic streaming, parallel execution, and structured tracking out-of-the-box. "
    "Furthermore, LangGraph has replaced the legacy AgentExecutor, offering stateful multi-actor agent coordination."
) * 5  # Duplicate to create a longer text

# Split text into documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(long_text)]
print(f"Split text into {len(docs)} document chunks.")


# --- 1. Stuff Summarization Chain ---
print("\n=== Stuff Summarization ===")

# Combine docs by joining page contents
combine_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
stuff_prompt = ChatPromptTemplate.from_template(
    "Write a concise summary of the following text:\n\n{text}"
)
stuff_chain = {"text": combine_docs} | stuff_prompt | llm | StrOutputParser()

stuff_summary = stuff_chain.invoke(docs)
print(stuff_summary)


# --- 2. Map-Reduce Summarization Chain ---
print("\n=== Map-Reduce Summarization ===")

# Step A: Map - Summarize each chunk individually
map_prompt = ChatPromptTemplate.from_template(
    "Summarize this specific section of text:\n\n{context}"
)
map_chain = map_prompt | llm | StrOutputParser()

# We convert our list of Document objects to a list of dict inputs expected by map_prompt
prepare_map_inputs = RunnableLambda(lambda docs: [{"context": d.page_content} for d in docs])
# .map() runs the map_chain over each element of the input list
map_step = prepare_map_inputs | map_chain.map()

# Step B: Reduce - Combine all individual summaries and run a final summarize prompt
reduce_prompt = ChatPromptTemplate.from_template(
    "The following are summaries of different parts of a document:\n\n"
    "{summaries}\n\n"
    "Based on the above, write a cohesive, comprehensive overall summary."
)
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Format the list of mapped summaries into the reduction prompt
reduce_step = RunnableLambda(lambda summaries: {"summaries": "\n\n".join(summaries)}) | reduce_chain

# The full Map-Reduce pipeline
map_reduce_chain = map_step | reduce_step

map_reduce_summary = map_reduce_chain.invoke(docs)
print(map_reduce_summary)
