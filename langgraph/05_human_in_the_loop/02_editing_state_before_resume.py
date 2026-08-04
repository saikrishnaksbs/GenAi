"""
Editing State Mid-Run Before Resuming
=========================================
Pausing a graph (via interrupt_before/after) is only half of a
human-in-the-loop workflow - the other half is letting a human *change*
the state before execution continues. `graph.update_state(config, values)`
writes new values into the checkpoint as if a node had produced them
(passing them through any reducers), creating a new checkpoint. The next
`.invoke(None, config)` call resumes from that edited state.
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
    return {"sent": True}


builder = StateGraph(State)
builder.add_node("draft_node", draft_node)
builder.add_node("send_node", send_node)
builder.add_edge(START, "draft_node")
builder.add_edge("draft_node", "send_node")
builder.add_edge("send_node", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["send_node"])

config = {"configurable": {"thread_id": "email-1"}}
graph.invoke({"draft_email": "", "recipient": "team@example.com", "sent": False}, config=config)

# Graph is paused before send_node. A human edits the draft to add detail.
current = graph.get_state(config).values
print(current["draft_email"])
# -> "Hey, quick reminder about tomorrow's meeting."

# update_state merges these values into the checkpoint, exactly like a node
# return value would -- overwriting `draft_email` for the next resume.
graph.update_state(
    config,
    {"draft_email": current["draft_email"] + " Location: Conference Room B."},
)

# Resume from the edited checkpoint.
final = graph.invoke(None, config=config)
print(final["draft_email"])
# -> "Hey, quick reminder about tomorrow's meeting. Location: Conference Room B."
print(final["sent"])
# -> True
