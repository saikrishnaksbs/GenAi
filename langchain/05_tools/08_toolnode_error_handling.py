"""
EXPLAINING LANGGRAPH'S TOOLNODE FOR ERROR HANDLING
===================================================
In LangGraph, `ToolNode` is a prebuilt graph node designed to execute tools 
requested by a model. 

When building production agents, you must expect tool executions to fail 
occasionally (e.g., database timeout, invalid inputs, division by zero). 

Instead of crashing the program, `ToolNode` catches these exceptions, packages 
them into a `ToolMessage`, and appends them to the graph's messages state. 
This allows the LLM to see the error and try to correct itself.

This script demonstrates:
1. Setting up a tool that throws exceptions.
2. How the LLM initiates tool calls.
3. How `ToolNode` executes the tools, intercepts the failure, and returns the error message.
"""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode


# 1. Define a tool that throws exceptions on invalid inputs
@tool
def fetch_user_balance(account_id: str) -> float:
    """Fetch the account balance for a given alphanumeric account ID."""
    # Simulate a validation check that raises an exception
    if not account_id.isalnum():
        raise ValueError(
            f"Invalid Account ID '{account_id}': Must be alphanumeric."
        )
    
    # Mock database lookup
    balances = {"acc123": 450.75, "acc456": 1200.00}
    if account_id not in balances:
        raise KeyError(f"Account ID '{account_id}' not found.")
        
    return balances[account_id]


# 2. Instantiate ToolNode with the tool list
# ToolNode implements standard error-catching out of the box.
tool_node = ToolNode([fetch_user_balance])

# 3. Instantiate the Chat Model
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
model_with_tools = model.bind_tools([fetch_user_balance])

print("--- Step 1: LLM Tool Selection ---")
# Prompt containing an invalid account ID (has spaces, which violates .isalnum())
messages = [HumanMessage("What is the balance of account 'acc 123'?")]
ai_message = model_with_tools.invoke(messages)

print(f"Model requested tool calls:\n{ai_message.tool_calls}\n")


print("--- Step 2: Executing via ToolNode (Catching Errors) ---")
# Invoke the ToolNode. We pass the current graph state containing our messages.
# ToolNode reads the last message (which is the AIMessage requesting the tool call).
node_result = tool_node.invoke({"messages": messages + [ai_message]})

# Inspect the result returned by ToolNode
print("ToolNode Output State:")
for idx, msg in enumerate(node_result["messages"]):
    print(f"Message {idx + 1} (Type: {type(msg).__name__}):")
    print(f"  Tool Call ID: {msg.tool_call_id}")
    print(f"  Content:      {msg.content}\n")

# Notice how ToolNode caught the ValueError and returned it as an error message 
# in the content block instead of throwing the exception and crashing the runtime.
