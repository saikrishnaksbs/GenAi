"""
The interrupt() Function: Dynamic Interrupts
================================================
Static `interrupt_before`/`interrupt_after` pause at fixed node
boundaries decided at compile time. The newer `interrupt()` function
(from `langgraph.types`) instead lets a NODE decide, at runtime, to pause
mid-execution and surface a payload to the caller - e.g. "here's the
data I need a human to review." Resuming passes a `Command(resume=...)`
back in, which becomes interrupt()'s return value inside the node,
letting execution continue with the human's input woven directly into
the node's logic.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    order_total: float
    approval_note: str
    status: str


def review_order_node(state: State) -> dict:
    if state["order_total"] > 1000:
        # interrupt() pauses execution HERE and surfaces this payload to
        # the caller of .invoke()/.stream(). The graph's checkpoint captures
        # exactly this point, so it can resume later with a human's answer.
        human_response = interrupt(
            {
                "question": f"Approve order totaling ${state['order_total']}?",
                "order_total": state["order_total"],
            }
        )
        # Whatever is passed via Command(resume=...) becomes the return
        # value of interrupt() when execution resumes.
        return {"status": human_response["decision"], "approval_note": human_response.get("note", "")}
    return {"status": "auto-approved"}


builder = StateGraph(State)
builder.add_node("review_order", review_order_node)
builder.add_edge(START, "review_order")
builder.add_edge("review_order", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "order-99"}}
result = graph.invoke({"order_total": 1500.0, "approval_note": "", "status": ""}, config=config)

# When an interrupt() fires, invoke() returns early with an `__interrupt__`
# key describing the pause instead of running to completion.
print(result.get("__interrupt__"))
# -> (Interrupt(value={'question': 'Approve order totaling $1500.0?', 'order_total': 1500.0}, ...),)

# Resume by invoking with a Command(resume=...) carrying the human's answer.
# This value flows straight back as interrupt()'s return value in the node.
final = graph.invoke(
    Command(resume={"decision": "approved", "note": "Verified with customer by phone."}),
    config=config,
)
print(final)
# -> {"order_total": 1500.0, "approval_note": "Verified with customer by phone.", "status": "approved"}
