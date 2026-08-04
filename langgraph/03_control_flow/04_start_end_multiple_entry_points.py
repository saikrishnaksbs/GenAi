"""
START and END Sentinels, Multiple Entry Points
===================================================
`START` and `END` are special sentinel nodes (not real functions) that
mark where execution begins and terminates. Every graph needs at least
one edge from START and one path reaching END. A graph isn't limited to
a single entry point: a conditional edge attached to START can route the
very first step to different nodes depending on the input, effectively
giving the graph multiple "entry points" chosen dynamically.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    mode: str
    input_data: str
    output: str


def fast_path_node(state: State) -> dict:
    return {"output": f"[fast] {state['input_data']}"}


def thorough_path_node(state: State) -> dict:
    return {"output": f"[thorough, validated] {state['input_data']}"}


def finalize_node(state: State) -> dict:
    return {"output": state["output"] + " -- done"}


def choose_entry(state: State) -> Literal["fast_path_node", "thorough_path_node"]:
    # A conditional edge off START picks the entry point based on input,
    # instead of every invocation always starting at the same node.
    return "fast_path_node" if state["mode"] == "fast" else "thorough_path_node"


builder = StateGraph(State)
builder.add_node("fast_path_node", fast_path_node)
builder.add_node("thorough_path_node", thorough_path_node)
builder.add_node("finalize_node", finalize_node)

# Conditional entry point: START routes dynamically instead of a fixed add_edge.
builder.add_conditional_edges(START, choose_entry)
builder.add_edge("fast_path_node", "finalize_node")
builder.add_edge("thorough_path_node", "finalize_node")
builder.add_edge("finalize_node", END)  # Both paths converge and reach END.

graph = builder.compile()

print(graph.invoke({"mode": "fast", "input_data": "hello", "output": ""}))
# -> {..., "output": "[fast] hello -- done"}

print(graph.invoke({"mode": "careful", "input_data": "hello", "output": ""}))
# -> {..., "output": "[thorough, validated] hello -- done"}
