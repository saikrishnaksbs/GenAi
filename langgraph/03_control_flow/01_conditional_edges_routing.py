"""
add_conditional_edges with a Routing Function
=================================================
This is the core branching primitive in LangGraph. After a source node
runs, a routing function inspects the resulting state and returns the
name (or names) of the node(s) to execute next. `add_conditional_edges`
wires that function into the graph. It's commonly used to check whether
an LLM emitted tool calls, whether validation passed, or - as here -
whether a customer request needs escalation.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    request: str
    category: str
    response: str


def classify_request(state: State) -> dict:
    text = state["request"].lower()
    if "refund" in text or "cancel" in text:
        category = "billing"
    elif "bug" in text or "error" in text:
        category = "technical"
    else:
        category = "general"
    return {"category": category}


def billing_node(state: State) -> dict:
    return {"response": "Routed to billing team."}


def technical_node(state: State) -> dict:
    return {"response": "Routed to technical support."}


def general_node(state: State) -> dict:
    return {"response": "Routed to general inquiries."}


def route_by_category(state: State) -> Literal["billing_node", "technical_node", "general_node"]:
    # The routing function's return value must match a key in the mapping
    # passed to add_conditional_edges (or a real node name if no mapping given).
    return f"{state['category']}_node" if state["category"] != "general" else "general_node"


builder = StateGraph(State)
builder.add_node("classify", classify_request)
builder.add_node("billing_node", billing_node)
builder.add_node("technical_node", technical_node)
builder.add_node("general_node", general_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category)
builder.add_edge("billing_node", END)
builder.add_edge("technical_node", END)
builder.add_edge("general_node", END)

graph = builder.compile()

print(graph.invoke({"request": "I want a refund", "category": "", "response": ""}))
# -> {..., "category": "billing", "response": "Routed to billing team."}
