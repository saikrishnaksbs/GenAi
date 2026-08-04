"""
Agent Handoffs via Command(goto=...)
========================================
`Command` lets a node combine a state update AND a routing decision in
one return value, instead of using a separate conditional edge function.
Returning `Command(update={...}, goto="other_node")` from a node updates
state and immediately hands off control to `other_node` - a natural fit
for multi-agent systems where any agent might decide, based on its own
reasoning, to hand off directly to a specific peer agent rather than
going through a central router every time.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.types import Command


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def intake_agent(state: State) -> Command[Literal["billing_agent", "tech_agent"]]:
    text = state["messages"][-1].content.lower()
    note = AIMessage(content="Intake: routing your request.")
    if "invoice" in text or "payment" in text:
        # Command bundles the state update with the handoff target -- no
        # separate add_conditional_edges call needed for this decision.
        return Command(update={"messages": [note]}, goto="billing_agent")
    return Command(update={"messages": [note]}, goto="tech_agent")


def billing_agent(state: State) -> dict:
    return {"messages": [AIMessage(content="Billing agent: I can help with that invoice.")]}


def tech_agent(state: State) -> dict:
    return {"messages": [AIMessage(content="Tech agent: let's debug this together.")]}


builder = StateGraph(State)
builder.add_node("intake_agent", intake_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("tech_agent", tech_agent)

builder.add_edge(START, "intake_agent")
# No add_conditional_edges call for intake_agent's routing -- Command(goto=...)
# inside the node itself declares the destination(s) at runtime.
builder.add_edge("billing_agent", END)
builder.add_edge("tech_agent", END)

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="I have a question about my invoice")]})
for m in result["messages"]:
    print(f"{m.__class__.__name__}: {m.content}")
# -> HumanMessage: I have a question about my invoice
# -> AIMessage: Intake: routing your request.
# -> AIMessage: Billing agent: I can help with that invoice.
