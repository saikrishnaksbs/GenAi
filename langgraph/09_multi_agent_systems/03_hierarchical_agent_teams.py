"""
Hierarchical Agent Teams: A Supervisor of Supervisors
==========================================================
Complex systems compose the supervisor pattern recursively: a top-level
supervisor delegates not to individual workers but to TEAM supervisors,
each of which is itself a compiled subgraph managing its own workers.
Because a compiled graph is just a Runnable, a "team" subgraph can be
added as a single node in the top-level graph, exactly like a leaf
worker would be - the top-level supervisor doesn't need to know the team
has internal structure.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class TeamState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str


# --- Engineering team: its own internal supervisor + workers ---
def eng_supervisor(state: TeamState) -> dict:
    text = state["messages"][-1].content.lower()
    return {"next_worker": "frontend_dev" if "ui" in text else "backend_dev"}


def frontend_dev(state: TeamState) -> dict:
    return {"messages": [AIMessage(content="Frontend dev: updated the UI component.")]}


def backend_dev(state: TeamState) -> dict:
    return {"messages": [AIMessage(content="Backend dev: fixed the API endpoint.")]}


def route_eng_team(state: TeamState) -> Literal["frontend_dev", "backend_dev"]:
    return state["next_worker"]


eng_builder = StateGraph(TeamState)
eng_builder.add_node("eng_supervisor", eng_supervisor)
eng_builder.add_node("frontend_dev", frontend_dev)
eng_builder.add_node("backend_dev", backend_dev)
eng_builder.add_edge(START, "eng_supervisor")
eng_builder.add_conditional_edges("eng_supervisor", route_eng_team)
eng_builder.add_edge("frontend_dev", END)
eng_builder.add_edge("backend_dev", END)
engineering_team = eng_builder.compile()  # A whole team, exposed as one Runnable.


# --- Top-level supervisor: delegates to TEAMS, not individual workers ---
class TopState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_team: str


def top_supervisor(state: TopState) -> dict:
    return {"next_team": "engineering_team"}  # Only one team exists in this demo.


top_builder = StateGraph(TopState)
top_builder.add_node("top_supervisor", top_supervisor)
# The engineering team's compiled subgraph is added as a single node --
# the top-level graph treats it as an opaque worker.
top_builder.add_node("engineering_team", engineering_team)
top_builder.add_edge(START, "top_supervisor")
top_builder.add_edge("top_supervisor", "engineering_team")
top_builder.add_edge("engineering_team", END)

top_graph = top_builder.compile()

result = top_graph.invoke({
    "messages": [HumanMessage(content="Please improve the UI")],
    "next_team": "",
})
for m in result["messages"]:
    print(f"{m.__class__.__name__}: {m.content}")
# -> HumanMessage: Please improve the UI
# -> AIMessage: Frontend dev: updated the UI component.
