"""
Compiled Graph as a Subgraph Node in LangGraph
===============================================

Concept:
--------
In LangGraph, a compiled graph (`builder.compile()`) is a Runnable.
Because every node in a StateGraph accepts state and returns state updates,
a compiled graph can be passed directly as a node into a PARENT graph:

    parent_builder.add_node("research_team", research_subgraph)

Architecture Diagram:
--------------------
[PARENT GRAPH: ReportState]
 START ---> [Node: research_team (SUBGRAPH)] ---> [Node: write_report] ---> END
                 |
                 +--> [SUBGRAPH: ResearchState]
                      START ---> [Node: web_search] ---> [Node: summarize] ---> END

State Sharing & Key Matching:
-----------------------------
1. Parent passes its current state to the subgraph node.
2. Subgraph extracts keys matching `ResearchState` (`topic`).
3. Subgraph runs internally (`web_search` -> `summarize`).
4. Subgraph returns updates to parent (`findings`), which parent state merges.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# =====================================================================
# 1. SUBGRAPH DEFINITION (Self-contained Research Team)
# =====================================================================

class ResearchState(TypedDict):
    """Internal state schema for the research subgraph."""
    topic: str
    findings: str


def web_search_node(state: ResearchState) -> dict:
    print(f"  [SUBGRAPH: web_search] Searching sources for topic: '{state['topic']}'...")
    return {"findings": f"Found raw benchmark data for {state['topic']}"}


def summarize_node(state: ResearchState) -> dict:
    print(f"  [SUBGRAPH: summarize] Refining and summarizing raw findings...")
    refined = state["findings"] + " -> Extracted key performance stats."
    return {"findings": refined}


# Build & compile the self-contained subgraph
subgraph_builder = StateGraph(ResearchState)
subgraph_builder.add_node("web_search", web_search_node)
subgraph_builder.add_node("summarize", summarize_node)

subgraph_builder.add_edge(START, "web_search")
subgraph_builder.add_edge("web_search", "summarize")
subgraph_builder.add_edge("summarize", END)

# Compiled graph object (Runnable)
research_subgraph = subgraph_builder.compile()


# =====================================================================
# 2. PARENT GRAPH DEFINITION (Main Workflow)
# =====================================================================

class ReportState(TypedDict):
    """Outer / Parent state schema."""
    topic: str
    findings: str
    report: str


def write_report_node(state: ReportState) -> dict:
    print(f"[PARENT: write_report] Drafting final report from findings...")
    report_content = (
        f"=== FINAL REPORT: {state['topic']} ===\n"
        f"Summary of Research: {state['findings']}"
    )
    return {"report": report_content}


# Build parent graph
parent_builder = StateGraph(ReportState)

# Key step: Adding the compiled subgraph directly as a node in the parent graph
parent_builder.add_node("research_team", research_subgraph)
parent_builder.add_node("write_report", write_report_node)

parent_builder.add_edge(START, "research_team")
parent_builder.add_edge("research_team", "write_report")
parent_builder.add_edge("write_report", END)

parent_graph = parent_builder.compile()


# =====================================================================
# 3. EXECUTION DEMO
# =====================================================================

if __name__ == "__main__":
    print("=== Invoking Parent Graph ===")
    initial_state = {
        "topic": "LangGraph Subgraphs",
        "findings": "",
        "report": ""
    }
    
    final_output = parent_graph.invoke(initial_state)
    
    print("\n=== Execution Complete ===")
    print(final_output["report"])

