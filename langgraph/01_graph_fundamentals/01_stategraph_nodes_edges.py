"""
StateGraph, Nodes, and Edges
===============================
LangGraph models an application as a graph: `StateGraph` holds a shared
state schema, `nodes` are functions that read/update that state, and `edges`
define which node runs next. Unlike a linear LCEL chain, a graph can loop,
branch, and revisit nodes — which is what makes it suited for agents.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str


def research_node(state: State) -> dict:
    # Nodes are plain functions: take the state, return a partial update dict.
    return {"answer": f"Researching: {state['question']}"}


def summarize_node(state: State) -> dict:
    return {"answer": state["answer"] + " -> summarized."}


builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_node("summarize", summarize_node)

# Edges wire nodes together. START/END are special sentinel nodes marking
# the graph's entry and exit points.
builder.add_edge(START, "research")
builder.add_edge("research", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

result = graph.invoke({"question": "What is LangGraph?", "answer": ""})
print(result)
# -> {"question": "What is LangGraph?", "answer": "Researching: What is LangGraph? -> summarized."}

# You can also visualize the compiled graph structure:
print(graph.get_graph().draw_ascii())
