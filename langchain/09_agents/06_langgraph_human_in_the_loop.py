"""
HUMAN-IN-THE-LOOP AGENTS IN LANGGRAPH (Breakpoints & Manual Approval)
======================================================================
One of the major limitations of AgentExecutor was the inability to pause agent
execution to wait for human verification or feedback (e.g. before making a payment
or deleting a resource).

LangGraph introduces **Breakpoints** natively. When compiling a graph, you can
specify:
- `interrupt_before`: Pause execution *before* a specific node is run.
- `interrupt_after`: Pause execution *after* a specific node completes.

You can then inspect the state, approve/modify the input, and resume execution by
invoking the graph again with `None` as input on the same `thread_id`.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode

# Define a sensitive tool that requires human approval
@tool
def delete_database_entry(entry_id: str) -> str:
    """Delete a database record. Highly sensitive operation."""
    print(f"-> Executing Tool: Database Entry '{entry_id}' DELETED successfully.")
    return f"Database entry {entry_id} has been deleted."

tools = [delete_database_entry]
tool_node = ToolNode(tools)
model = ChatOllama(model="qwen2.5:1.5b").bind_tools(tools)

# 1. Build the graph
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

# 2. Compile the graph WITH a checkpointer AND interrupt_before sensitive tool node
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # Automatically pause execution before executing any tools
)

config = {"configurable": {"thread_id": "sensitive-db-session"}}
inputs = {"messages": [("user", "Delete database entry 'usr_9823'.")]}

# 3. Run the graph until the breakpoint
print("=== Starting Agent Execution ===")
for event in app.stream(inputs, config=config, stream_mode="values"):
    last_msg = event["messages"][-1]
    last_msg.pretty_print()

# Check where the graph is currently at
state = app.get_state(config)
print("\n--- Execution Paused ---")
print("Next Node to Execute:", state.next)
print("Pending Tool Calls requested by LLM:")
for tc in state.values["messages"][-1].tool_calls:
    print(f"- Tool: {tc['name']} with arguments: {tc['args']}")

# 4. Simulate human review
print("\n[Human Operator Decision]: Approved. Resuming execution...")

# To resume execution from the paused state, we invoke the app again passing None as input
for event in app.stream(None, config=config, stream_mode="values"):
    last_msg = event["messages"][-1]
    last_msg.pretty_print()

print("\n=== Agent Execution Finished ===")
