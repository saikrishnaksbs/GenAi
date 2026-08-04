"""
Approval Workflows: Approve/Reject a Tool Call
==================================================
A common human-in-the-loop pattern is pausing right before a sensitive
tool executes (e.g. sending money, deleting data), letting a human
approve or reject it, then routing accordingly. This combines
interrupt_before on a "gate" node with a conditional edge that reads a
human-supplied decision, set via update_state, after resume.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    action: str
    amount: float
    decision: str  # set by a human between pause and resume
    result: str


def propose_action(state: State) -> dict:
    return {"action": f"transfer ${state['amount']}", "decision": "pending"}


def await_approval(state: State) -> dict:
    # This node is a no-op "gate" -- its only purpose is to be the node the
    # graph pauses BEFORE, giving a human a chance to set `decision` via
    # update_state prior to the conditional edge evaluating it.
    return {}


def execute_action(state: State) -> dict:
    return {"result": f"Executed: {state['action']}"}


def reject_action(state: State) -> dict:
    return {"result": f"Rejected: {state['action']}"}


def route_on_decision(state: State) -> Literal["execute_action", "reject_action"]:
    return "execute_action" if state["decision"] == "approve" else "reject_action"


builder = StateGraph(State)
builder.add_node("propose_action", propose_action)
builder.add_node("await_approval", await_approval)
builder.add_node("execute_action", execute_action)
builder.add_node("reject_action", reject_action)

builder.add_edge(START, "propose_action")
builder.add_edge("propose_action", "await_approval")
builder.add_conditional_edges("await_approval", route_on_decision)
builder.add_edge("execute_action", END)
builder.add_edge("reject_action", END)

# Pause BEFORE the gate node, so the decision is set before routing occurs.
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["await_approval"])

config = {"configurable": {"thread_id": "txn-1"}}
graph.invoke({"action": "", "amount": 5000.0, "decision": "", "result": ""}, config=config)

snapshot = graph.get_state(config)
print(snapshot.next)
# -> ("await_approval",)  paused, waiting for a human decision

# A human reviews the proposed action and rejects it.
graph.update_state(config, {"decision": "reject"})

final = graph.invoke(None, config=config)
print(final["result"])
# -> "Rejected: transfer $5000.0"
