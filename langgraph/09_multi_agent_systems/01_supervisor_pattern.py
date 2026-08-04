"""
Supervisor Pattern: A Router Agent Delegating to Worker Agents
====================================================================
The supervisor pattern uses one "router" node - often an LLM call with
structured output - to decide which specialized worker node should
handle the current request, then loops back to the supervisor after each
worker runs so it can decide the next step (or finish). This keeps
worker agents simple and single-purpose while centralizing the
delegation logic.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str


def supervisor_node(state: State) -> dict:
    # In real code, this would be an LLM call with structured output, e.g.
    # llm.with_structured_output(RouteDecision).invoke(state["messages"])
    last_text = state["messages"][-1].content.lower()
    if "code" in last_text:
        route = "coder_agent"
    elif "search" in last_text or "find" in last_text:
        route = "researcher_agent"
    else:
        route = "FINISH"
    return {"next_worker": route}


def researcher_agent(state: State) -> dict:
    return {"messages": [AIMessage(content="Researcher: here's what I found.")]}


def coder_agent(state: State) -> dict:
    return {"messages": [AIMessage(content="Coder: here's the implementation.")]}


def route_from_supervisor(state: State) -> Literal["researcher_agent", "coder_agent", "__end__"]:
    return state["next_worker"] if state["next_worker"] != "FINISH" else "__end__"


builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher_agent", researcher_agent)
builder.add_node("coder_agent", coder_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {"researcher_agent": "researcher_agent", "coder_agent": "coder_agent", "__end__": END},
)
# Workers report back to the supervisor, which can loop or finish next.
builder.add_edge("researcher_agent", "supervisor")
builder.add_edge("coder_agent", "supervisor")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="Please write some code")], "next_worker": ""})
for m in result["messages"]:
    print(f"{m.__class__.__name__}: {m.content}")
# -> HumanMessage: Please write some code
# -> AIMessage: Coder: here's the implementation.
# (supervisor then routes to FINISH since the follow-up message doesn't
#  contain "code" or "search" keywords again)
