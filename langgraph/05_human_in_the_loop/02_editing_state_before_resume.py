"""
Editing State Mid-Run Before Resuming (Interactive Terminal Interrupt)
======================================================================
Pausing a graph (via interrupt_before/after) allows a human to *change*
the state before execution continues. `graph.update_state(config, values)`
writes new values into the checkpoint as if a node had produced them,
creating a new checkpoint. The next `.invoke(None, config)` call resumes
from that edited state.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    draft_email: str
    recipient: str
    sent: bool


def draft_node(state: State) -> dict:
    return {"draft_email": "Hey, quick reminder about tomorrow's meeting."}


def send_node(state: State) -> dict:
    print(f"\n[SENDING EMAIL] To: {state['recipient']}")
    print(f"Content: \"{state['draft_email']}\"")
    return {"sent": True}


builder = StateGraph(State)
builder.add_node("draft_node", draft_node)
builder.add_node("send_node", send_node)
builder.add_edge(START, "draft_node")
builder.add_edge("draft_node", "send_node")
builder.add_edge("send_node", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["send_node"])

config = {"configurable": {"thread_id": "email-1"}}

print("=== 1. Invoking Graph (Runs until send_node) ===")
graph.invoke({"draft_email": "", "recipient": "team@example.com", "sent": False}, config=config)

# Graph is paused before send_node. Inspect current snapshot.
current = graph.get_state(config).values
print(f"\n[INTERRUPT DETECTED] Paused before 'send_node'.")
print(f"Current Recipient: {current['recipient']}")
print(f"Current Draft: \"{current['draft_email']}\"")

# --- REAL TERMINAL INTERRUPT ---
print("\n--- HUMAN EDIT INTERFACE ---")
new_recipient = input(f"Enter recipient (press Enter to keep '{current['recipient']}'): ").strip()
if not new_recipient:
    new_recipient = current["recipient"]

user_addition = input("Add text to email draft (press Enter to keep unchanged): ").strip()
if user_addition:
    updated_draft = f"{current['draft_email']} {user_addition}"
else:
    updated_draft = current["draft_email"]

# update_state merges these values into the checkpoint
graph.update_state(
    config,
    {"draft_email": updated_draft, "recipient": new_recipient},
)
print("-> Checkpoint state updated by human.")

print("\n=== 2. Resuming Graph Execution ===")
final = graph.invoke(None, config=config)
print(f"Final Sent Status: {final['sent']}")

