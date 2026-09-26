"""
RECURSION LIMITS AND EARLY STOPPING IN LANGGRAPH (Replacing max_iterations)
=============================================================================
In legacy AgentExecutor, `max_iterations` and `early_stopping_method` kept agents from
looping indefinitely.

In LangGraph, you control this in two ways:
1. **Config Recursion Limit**: Pass a `recursion_limit` in the run configuration. If the
   graph execution takes more steps than the limit, it raises a `GraphRecursionError`.
2. **State Loop Counter**: Store a step count in the state and use a conditional edge to
   route to an early stopping node before the cap is hit.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

# Define a tool that returns a dummy response to simulate a loop
@tool
def search_web(query: str) -> str:
    """Simulate web search that fails to find definitive answers, forcing loops."""
    return "Result inconclusive, try searching again."

tools = [search_web]
tool_node = ToolNode(tools)
model = ChatOllama(model="qwen2.5:1.5b").bind_tools(tools)

# --- Approach 1: Graph Recursion Limit (Catching Error) ---
print("=== Approach 1: Config Recursion Limit ===")

# Build standard graph using prebuilt MessagesState
workflow = StateGraph(MessagesState)
workflow.add_node("agent", lambda state: {"messages": [model.invoke(state["messages"])]})
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")
agent = workflow.compile()

# Set recursion_limit to 5. Since each loop is agent -> tools -> agent (2 steps per loop),
# a limit of 5 restricts the execution to 2 loops.
config = {"recursion_limit": 5}
try:
    for event in agent.stream(
        {"messages": [("user", "What is the exact population of Mars colonies?")]},
        config=config,
        stream_mode="values"
    ):
        print(f"Step executed. Messages count: {len(event['messages'])}")
except GraphRecursionError:
    print("\n[STOPPED] Execution halted: hit the graph recursion limit.")


# --- Approach 2: State Loop Counter (Graceful Canned Response) ---
print("\n=== Approach 2: Graceful State-Based Early Stopping ===")

class CustomState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int

def call_model(state: CustomState):
    # Increment loop count and invoke model
    current_count = state.get("loop_count", 0) + 1
    response = model.invoke(state["messages"])
    return {"messages": [response], "loop_count": current_count}

def check_loop_limit(state: CustomState):
    # Stop early if we have run the model 3 times already
    if state.get("loop_count", 0) >= 3:
        return "early_stop"
    
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def early_stop_node(state: CustomState):
    # Return a canned fallback message instead of throwing an exception
    fallback = AIMessage(content="Agent execution halted: maximum search iterations reached without resolution.")
    return {"messages": [fallback]}

custom_workflow = StateGraph(CustomState)
custom_workflow.add_node("agent", call_model)
custom_workflow.add_node("tools", tool_node)
custom_workflow.add_node("early_stop", early_stop_node)

custom_workflow.set_entry_point("agent")
custom_workflow.add_conditional_edges(
    "agent",
    check_loop_limit,
    {
        "tools": "tools",
        "early_stop": "early_stop",
        END: END
    }
)
custom_workflow.add_edge("tools", "agent")
custom_workflow.add_edge("early_stop", END)

controlled_agent = custom_workflow.compile()

# Invoke the agent. It will stop gracefully and return our fallback message.
final_state = controlled_agent.invoke(
    {"messages": [("user", "What is the exact population of Mars colonies?")], "loop_count": 0}
)
print("\nFinal Answer:")
print(final_state["messages"][-1].content)
