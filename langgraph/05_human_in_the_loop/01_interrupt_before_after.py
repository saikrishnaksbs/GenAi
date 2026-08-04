"""
interrupt_before / interrupt_after: Pausing a Graph
========================================================
`compile(interrupt_before=[...])` / `interrupt_after=[...]` tell
LangGraph to pause execution right before (or after) specific named
nodes run, returning control to the caller instead of continuing. This
requires a checkpointer, since the graph's progress must be saved so it
can be resumed later. Resuming is done by calling `.invoke(None, config)`
with the same thread_id - passing `None` as input means "continue from
where we paused" rather than starting a new run.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    draft: str
    approved: bool
    published: str


def write_draft(state: State) -> dict:
    return {"draft": "Quarterly earnings look strong."}


def publish(state: State) -> dict:
    return {"published": f"PUBLISHED: {state['draft']}"}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("publish", publish)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "publish")
builder.add_edge("publish", END)

# Pause right before `publish` runs, so a human can review the draft first.
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["publish"])

config = {"configurable": {"thread_id": "post-1"}}
result = graph.invoke({"draft": "", "approved": False, "published": ""}, config=config)

# Execution stopped before `publish`. Inspect state to review the draft.
snapshot = graph.get_state(config)
print(snapshot.next)
# -> ("publish",)  -- the graph is paused right here, waiting to resume.
print(snapshot.values["draft"])
# -> "Quarterly earnings look strong."

# A human reviews the draft out-of-band, then resumes by invoking with
# input=None -- this tells LangGraph "continue the paused run," not "start over".
final = graph.invoke(None, config=config)
print(final["published"])
# -> "PUBLISHED: Quarterly earnings look strong."

# interrupt_after works the same way but pauses immediately AFTER the named
# node completes, e.g. interrupt_after=["write_draft"] would pause once the
# draft exists but before `publish` is even scheduled to run.
