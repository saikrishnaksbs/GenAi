"""
LCEL CHAIN TOPOLOGY & GRAPH VISUALIZATION (.GET_GRAPH())
======================================================
As LCEL chains grow in complexity (parallel execution, branching, data routing), 
it becomes hard to trace inputs, outputs, and connections. LangChain provides a 
native `.get_graph()` method on all Runnables to analyze their execution topologies.

This script demonstrates how to:
1. Construct a branched LCEL pipeline.
2. Call `.get_graph()` to retrieve the nodes and edges.
3. Print an ASCII representation of the chain sequence structure.
4. Export the topology as a Mermaid diagram code block.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

# 1. Build a branched LCEL pipeline:
# We receive text, run two parallel tasks (sentiment analysis and keyword extraction),
# and pipe their combined results to a final synthesizer.
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

sentiment_prompt = ChatPromptTemplate.from_template("Analyze the sentiment of this text: {text}")
keyword_prompt = ChatPromptTemplate.from_template("Extract the top 3 keywords from this text: {text}")

sentiment_branch = sentiment_prompt | model | StrOutputParser()
keyword_branch = keyword_prompt | model | StrOutputParser()

# Combine branches in parallel
parallel_branches = RunnableParallel(
    sentiment=sentiment_branch,
    keywords=keyword_branch
)

synthesizer_prompt = ChatPromptTemplate.from_template(
    "Synthesize the following context into a final report:\n"
    "Sentiment analysis results: {sentiment}\n"
    "Keywords extracted: {keywords}"
)

# Final complete chain
full_chain = (
    {"text": RunnablePassthrough()}
    | parallel_branches
    | synthesizer_prompt
    | model
    | StrOutputParser()
)


# 2. Get Graph & Visualize
graph = full_chain.get_graph()

print("--- Method 1: Print ASCII Graph Topology ---")
# print_ascii() renders a clean text drawing of the nodes and flow arrows
graph.print_ascii()


print("\n--- Method 2: Draw Mermaid Chart Code ---")
# draw_mermaid() outputs standard Mermaid diagram markup which can be rendered
# in markdown files or compatible editors.
mermaid_code = graph.draw_mermaid()
print(mermaid_code)
