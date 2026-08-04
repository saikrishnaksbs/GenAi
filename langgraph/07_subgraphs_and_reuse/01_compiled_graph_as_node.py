"""
Adding a Compiled Graph as a Subgraph Node
==============================================
A compiled `StateGraph` is itself just a Runnable, so you can add it
directly as a node in a *parent* graph via `add_node("name", compiled_subgraph)`.
When the parent and subgraph share the same state schema (or overlapping
keys), LangGraph automatically passes the relevant parts of the parent
state in and merges the subgraph's output back. This is the simplest way
to compose smaller graphs into a larger pipeline.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# --- Subgraph: a self-contained "research" pipeline ---
class ResearchState(TypedDict):
    topic: str
    findings: str


def search_node(state: ResearchState) -> dict:
    return {"findings": f"raw findings about {state['topic']}"}


def refine_node(state: ResearchState) -> dict:
    return {"findings": state["findings"] + " (refined)"}


research_builder = StateGraph(ResearchState)
research_builder.add_node("search", search_node)
research_builder.add_node("refine", refine_node)
research_builder.add_edge(START, "search")
research_builder.add_edge("search", "refine")
research_builder.add_edge("refine", END)
research_subgraph = research_builder.compile()  # A compiled graph = a Runnable.


# --- Parent graph: shares the `topic`/`findings` keys with the subgraph ---
class ReportState(TypedDict):
    topic: str
    findings: str
    report: str


def write_report(state: ReportState) -> dict:
    return {"report": f"Report on {state['topic']}: {state['findings']}"}


parent_builder = StateGraph(ReportState)
# The compiled subgraph is added exactly like any node function.
parent_builder.add_node("research", research_subgraph)
parent_builder.add_node("write_report", write_report)
parent_builder.add_edge(START, "research")
parent_builder.add_edge("research", "write_report")
parent_builder.add_edge("write_report", END)
parent_graph = parent_builder.compile()

result = parent_graph.invoke({"topic": "LangGraph", "findings": "", "report": ""})
print(result["report"])
# -> "Report on LangGraph: raw findings about LangGraph (refined)"
