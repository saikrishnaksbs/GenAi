"""
LANGGRAPH PERSISTENCE AND THREAD MEMORY (Modern replacement for ConversationMemory)
===================================================================================
In legacy LangChain, conversation history was managed via `ConversationBufferMemory`
or similar classes inside `AgentExecutor`.

In LangGraph, memory is handled via **Checkpointers** (like `MemorySaver` in-memory
checkpointing, or Postgres/Redis savers for production). 

When you compile a graph with a checkpointer, it automatically tracks state snapshots.
By passing a `thread_id` in the run config, LangGraph loads the corresponding state,
allowing you to resume multi-turn chat sessions natively across separate invocations.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Define a tool
@tool
def calculate_tax(income: int) -> float:
    """Calculate flat tax rate of 20% on income."""
    return income * 0.20

tools = [calculate_tax]
model = ChatOllama(model="qwen2.5:1.5b")

# Initialize checkpointer (MemorySaver)
checkpointer = MemorySaver()

# Compile the agent graph WITH the checkpointer
agent_with_memory = create_react_agent(
    model,
    tools,
    checkpointer=checkpointer
)

# --- Conversation Session 1: Thread A ---
print("=== Starting Conversation for Thread A (User: Bob) ===")
config_bob = {"configurable": {"thread_id": "thread-bob-session"}}

# Initial prompt
resp1 = agent_with_memory.invoke(
    {"messages": [("user", "Hi! My name is Bob and my annual income is $80,000.")]},
    config=config_bob
)
print("Bob 1:", resp1["messages"][-1].content)

# Follow-up prompt (relying on memory for both name and calculation)
resp2 = agent_with_memory.invoke(
    {"messages": [("user", "What is my name, and how much tax do I owe?")]},
    config=config_bob
)
print("\nBob 2:", resp2["messages"][-1].content)


# --- Conversation Session 2: Thread B (Different Thread) ---
print("\n=== Starting Conversation for Thread B (User: Alice) ===")
config_alice = {"configurable": {"thread_id": "thread-alice-session"}}

# Alice asks a question. She does not share Bob's state.
resp3 = agent_with_memory.invoke(
    {"messages": [("user", "What is my name?")]},
    config=config_alice
)
print("Alice 1:", resp3["messages"][-1].content)
