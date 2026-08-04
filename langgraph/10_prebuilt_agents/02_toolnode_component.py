"""
ToolNode: The Prebuilt Tool-Execution Component
====================================================
`ToolNode` is the building block `create_react_agent` uses internally
for executing tool calls. Given a list of tools, it inspects the latest
AIMessage in state for `tool_calls`, invokes each matching tool, and
appends the results as `ToolMessage`s - handling multiple parallel tool
calls, exceptions (turned into error ToolMessages by default), and tool
name lookup automatically. Use it directly when you want a react-style
loop but with custom routing logic instead of create_react_agent's
default wiring.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


@tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools = [word_count]
llm = ChatOllama(model="qwen2.5:1.5b").bind_tools(tools)

# ToolNode wraps the tool list: it reads state["messages"][-1].tool_calls
# and executes each one, producing ToolMessage(s) keyed by tool_call_id.
tool_node = ToolNode(tools)


def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def has_tool_calls(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "__end__"


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)  # Just a node -- no custom code needed to run tools.
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", has_tool_calls, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "agent")  # Loop back so the LLM sees the tool's result.

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="How many words are in 'the quick brown fox'?")]})
print(result["messages"][-1].content)
# -> "There are 4 words in 'the quick brown fox'."
