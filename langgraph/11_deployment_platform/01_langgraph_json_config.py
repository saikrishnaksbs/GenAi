"""
LangGraph Platform / langgraph.json Config for Deployment
===============================================================
To deploy a graph to LangGraph Platform (or run it locally with
`langgraph dev`/`langgraph up`), a project needs a `langgraph.json`
manifest at its root describing where compiled graphs live, dependencies,
and environment variables. This file itself contains the Python graph
definition that `langgraph.json` would point to, plus (as a string, for
reference) what that manifest typically looks like.

Expected `langgraph.json` alongside this project (not executed here,
just documented):

{
  "dependencies": ["."],
  "graphs": {
    "support_agent": "./11_deployment_platform/01_langgraph_json_config.py:graph"
  },
  "env": ".env"
}

The "graphs" mapping's value is "path/to/module.py:variable_name" -- the
platform imports that module and serves whatever compiled graph object
is bound to `variable_name`.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def support_node(state: State) -> dict:
    return {"messages": [AIMessage(content="How can I help you today?")]}


builder = StateGraph(State)
builder.add_node("support", support_node)
builder.add_edge(START, "support")
builder.add_edge("support", END)

# `graph` is the module-level compiled graph object that langgraph.json's
# "graphs" entry (support_agent -> this_file:graph) would point to.
# NOTE: LangGraph Platform manages its own persistence layer in the cloud,
# so a checkpointer is usually NOT passed here -- the platform injects one.
graph = builder.compile()

EXAMPLE_LANGGRAPH_JSON = """
{
  "dependencies": ["."],
  "graphs": {
    "support_agent": "./11_deployment_platform/01_langgraph_json_config.py:graph"
  },
  "env": ".env"
}
"""
print(EXAMPLE_LANGGRAPH_JSON)
