"""
LangGraph Studio for Visual Debugging
=========================================
LangGraph Studio is a visual IDE (desktop app, or browser UI served by
`langgraph dev`) that connects to a locally running graph server defined
by `langgraph.json`. It renders the graph's node/edge structure, lets you
trigger runs with custom input, watch state update live node-by-node,
inspect each checkpoint, and edit state mid-run to test human-in-the-loop
branches interactively -- all without writing driver code. This file
shows the kind of graph you'd point Studio at, with comments on what
Studio surfaces for each part.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, AIMessage


class TriageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    severity: str


def triage_node(state: TriageState) -> dict:
    # Studio's graph view shows this node highlighted while it's executing,
    # and the "Threads" panel lets you replay this exact run later.
    text = state["messages"][-1].content.lower()
    severity = "high" if "urgent" in text else "low"
    return {"severity": severity}


def escalate_node(state: TriageState) -> dict:
    return {"messages": [AIMessage(content="Escalated to on-call engineer.")]}


def queue_node(state: TriageState) -> dict:
    return {"messages": [AIMessage(content="Added to the regular support queue.")]}


def route_by_severity(state: TriageState) -> Literal["escalate_node", "queue_node"]:
    return "escalate_node" if state["severity"] == "high" else "queue_node"


builder = StateGraph(TriageState)
builder.add_node("triage", triage_node)
builder.add_node("escalate_node", escalate_node)
builder.add_node("queue_node", queue_node)
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_by_severity)
builder.add_edge("escalate_node", END)
builder.add_edge("queue_node", END)

# Studio requires a checkpointer to show "Threads" (past runs) and to
# support pausing/editing state interactively in its UI.
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["escalate_node"])

# To use Studio locally:
#   1. `pip install langgraph-cli`
#   2. Ensure langgraph.json points "graphs" at this module's `graph` variable.
#   3. Run `langgraph dev` from the project root.
#   4. Studio opens in-browser, connected to the local dev server, showing
#      this graph's nodes/edges and letting you invoke it with sample input,
#      watch the pause at "escalate_node", and inspect/edit `severity`
#      before resuming -- all through the UI, no extra script needed.
