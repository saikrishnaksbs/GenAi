"""
Parallel Node Execution: Fan-Out / Fan-In
=============================================
A single node can have multiple outgoing edges to independent downstream
nodes. LangGraph executes all of them concurrently within the same
"super-step" (fan-out), then waits for all of them to finish before the
next node that depends on their outputs runs (fan-in). This is how you
parallelize independent work - e.g. calling three retrieval sources at
once - without manual threading. Because multiple nodes write to state
in the same step, any shared keys they both update need a reducer to
avoid conflicts.
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    # Reducer required: three parallel nodes will each append to this list
    # in the same super-step, so LangGraph needs to know how to combine them.
    results: Annotated[list[str], operator.add]
    summary: str


def search_web(state: State) -> dict:
    return {"results": [f"web result for '{state['query']}'"]}


def search_docs(state: State) -> dict:
    return {"results": [f"docs result for '{state['query']}'"]}


def search_db(state: State) -> dict:
    return {"results": [f"db result for '{state['query']}'"]}


def summarize(state: State) -> dict:
    # Runs only after ALL three search nodes above have completed (fan-in).
    return {"summary": f"Combined {len(state['results'])} results."}


builder = StateGraph(State)
builder.add_node("search_web", search_web)
builder.add_node("search_docs", search_docs)
builder.add_node("search_db", search_db)
builder.add_node("summarize", summarize)

# Fan-out: START branches into three parallel nodes.
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_docs")
builder.add_edge(START, "search_db")

# Fan-in: summarize waits for all three to finish before running.
builder.add_edge("search_web", "summarize")
builder.add_edge("search_docs", "summarize")
builder.add_edge("search_db", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

result = graph.invoke({"query": "langgraph", "results": [], "summary": ""})
print(sorted(result["results"]))
# -> ["db result for 'langgraph'", "docs result for 'langgraph'", "web result for 'langgraph'"]
print(result["summary"])
# -> "Combined 3 results."
