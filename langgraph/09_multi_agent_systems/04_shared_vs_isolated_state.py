"""
Shared vs. Isolated State Between Agents in a Multi-Agent Graph
====================================================================
Multi-agent graphs can share a single state object (all agents read/write
the same `messages` list, seeing each other's full history) or keep each
agent's working state isolated, exposing only a summary to the rest of
the system. Shared state is simpler and gives agents full context;
isolated state reduces prompt bloat/cross-talk and is preferable when
agents have very different jobs and don't need each other's raw
scratch-work.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


# --- Pattern A: fully shared state -- every agent sees the whole conversation ---
class SharedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent_x_shared(state: SharedState) -> dict:
    # Sees every prior message, including other agents' internal chatter.
    return {"messages": [AIMessage(content=f"Agent X saw {len(state['messages'])} messages.")]}


def agent_y_shared(state: SharedState) -> dict:
    return {"messages": [AIMessage(content=f"Agent Y saw {len(state['messages'])} messages.")]}


shared_builder = StateGraph(SharedState)
shared_builder.add_node("agent_x", agent_x_shared)
shared_builder.add_node("agent_y", agent_y_shared)
shared_builder.add_edge(START, "agent_x")
shared_builder.add_edge("agent_x", "agent_y")
shared_builder.add_edge("agent_y", END)
shared_graph = shared_builder.compile()

print(shared_graph.invoke({"messages": [HumanMessage(content="start")]})["messages"][-1].content)
# -> "Agent Y saw 2 messages."  (sees the human message AND agent X's message)


# --- Pattern B: isolated state -- each agent runs its own private subgraph
#     scratchpad and only a distilled summary crosses back into shared state ---
class IsolatedParentState(TypedDict):
    task: str
    agent_x_summary: str
    agent_y_summary: str


def agent_x_isolated(state: IsolatedParentState) -> dict:
    # Does private multi-step work internally (not modeled here for brevity)
    # but only returns a short summary -- its scratch-work never enters
    # shared state, so agent_y never sees agent_x's raw reasoning.
    return {"agent_x_summary": f"X finished analyzing: {state['task']}"}


def agent_y_isolated(state: IsolatedParentState) -> dict:
    # Only receives the SUMMARY, not agent_x's full internal trace.
    return {"agent_y_summary": f"Y built on: {state['agent_x_summary']}"}


isolated_builder = StateGraph(IsolatedParentState)
isolated_builder.add_node("agent_x", agent_x_isolated)
isolated_builder.add_node("agent_y", agent_y_isolated)
isolated_builder.add_edge(START, "agent_x")
isolated_builder.add_edge("agent_x", "agent_y")
isolated_builder.add_edge("agent_y", END)
isolated_graph = isolated_builder.compile()

result = isolated_graph.invoke({"task": "market analysis", "agent_x_summary": "", "agent_y_summary": ""})
print(result["agent_y_summary"])
# -> "Y built on: X finished analyzing: market analysis"
