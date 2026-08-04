"""
tools_condition: The Prebuilt Routing Helper
================================================
Rather than hand-writing a routing function that checks
`state["messages"][-1].tool_calls`, `langgraph.prebuilt.tools_condition`
provides that exact check ready-made. Pass it to `add_conditional_edges`
and it returns the literal string "tools" when the last AIMessage
requested a tool call, or "__end__" otherwise - matching ToolNode's
default node name so the two prebuilts snap together with almost no
boilerplate.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo"}
    return capitals.get(country.lower(), "Unknown")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools = [lookup_capital]
llm = ChatOllama(model="qwen2.5:1.5b").bind_tools(tools)


def agent_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
# tools_condition replaces a hand-written routing function entirely: it
# returns "tools" when tool_calls are present, "__end__" otherwise.
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="What's the capital of Japan?")]})
print(result["messages"][-1].content)
# -> "The capital of Japan is Tokyo."

# This exact agent/tools/tools_condition/ToolNode wiring is precisely what
# create_react_agent builds for you under the hood -- useful to know when
# you need to deviate slightly from its defaults (e.g. custom pre/post nodes).
