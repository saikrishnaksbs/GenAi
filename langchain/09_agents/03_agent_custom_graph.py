"""
CUSTOM STATE GRAPH AGENT (Replacing AgentAction, AgentFinish, and Scratchpads)
==============================================================================
In legacy LangChain, custom agents required manually handling `AgentAction`,
`AgentFinish`, and formatting the `agent_scratchpad`.

In LangGraph, you represent the loop directly as a graph:
1. Define a state structure (typically subclassing TypedDict).
2. Define nodes (functions) that modify the state.
3. Define edges (control flow) that determine transitions based on the state.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. Define State: This carries all messages throughout the graph execution.
# `add_messages` is a reducer that appends new messages instead of overwriting.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define Tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 25°C and sunny."

tools = [get_weather]
tool_node = ToolNode(tools)

# Bind tools to model
model = ChatOllama(model="qwen2.5:1.5b").bind_tools(tools)

# 3. Define Graph Nodes
def call_model(state: AgentState):
    """Call the LLM and return the generated message."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# 4. Define Conditional Edge Logic (replaces AgentAction/AgentFinish checks)
def should_continue(state: AgentState):
    """Determine whether to route to the tool node or terminate."""
    last_message = state["messages"][-1]
    # If the LLM requested a tool call, transition to the tool node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, stop (END)
    return END

# 5. Build the StateGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edge from agent node
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Add normal edge from tools back to agent (to evaluate the tool output)
workflow.add_edge("tools", "agent")

# Compile graph
custom_agent = workflow.compile()

# 6. Execute the Custom Graph
print("=== Running Custom StateGraph Agent ===")
inputs = {"messages": [("human", "What is the weather in Delhi?")]}

for event in custom_agent.stream(inputs, stream_mode="values"):
    for message in event.get("messages", []):
        message.pretty_print()
