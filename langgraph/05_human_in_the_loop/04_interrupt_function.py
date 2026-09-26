"""
The interrupt() Function: Dynamic Interrupts (Interactive Terminal Interrupt)
=============================================================================
Static `interrupt_before`/`interrupt_after` pause at fixed node
boundaries decided at compile time. The dynamic `interrupt()` function
(from `langgraph.types`) instead lets a NODE decide, at runtime, to pause
mid-execution and surface a payload to the caller - e.g. "here's the
data I need a human to review." Resuming passes a `Command(resume=...)`
back in, which becomes interrupt()'s return value inside the node.
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
        # the caller. The graph's checkpoint captures exactly this point,
        # so it can resume later with a human's answer.
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

print("=== 1. Invoking Graph (Runs until dynamic interrupt() inside node) ===")
graph.invoke({"order_total": 1500.0, "approval_note": "", "status": ""}, config=config)

# Check state snapshot to see if any tasks produced an interrupt
snapshot = graph.get_state(config)
pending_tasks = snapshot.tasks

if pending_tasks and pending_tasks[0].interrupts:
    intr = pending_tasks[0].interrupts[0]
    payload = intr.value
    print(f"\n[INTERRUPT DETECTED] Surfaced Payload: {payload}")
    print(f"Question from Node: \"{payload.get('question')}\"")
    
    # --- REAL TERMINAL INTERRUPT ---
    print("\n--- HUMAN IN THE LOOP DYNAMIC INTERRUPT ---")
    decision_input = input("Decision [(a)pprove / (r)eject]: ").strip().lower()
    decision = "approved" if decision_input in ["a", "approve", "yes"] else "rejected"
    note = input("Enter approval note (optional): ").strip()
    
    print("\n=== 2. Resuming Graph Execution with Command(resume=...) ===")
    # Resume by invoking with a Command(resume=...) carrying the human's answer.
    # This value flows straight back as interrupt()'s return value in the node.
    final = graph.invoke(
        Command(resume={"decision": decision, "note": note}),
        config=config,
    )
    print(f"Final Graph State: {final}")


