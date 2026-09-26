"""
BASESTORE & INMEMORYSTORE: CROSS-THREAD LONG-TERM MEMORY
======================================================
In LangGraph:
- Checkpointers save state history for a SPECIFIC conversational session (thread_id).
- Stores (subclassing `BaseStore`, like `InMemoryStore`) persist memories 
  ACROSS MULTIPLE THREADS (e.g., storing user preferences, profiles, or custom facts).

This script demonstrates using `InMemoryStore` to build a personalization agent:
1. Read user preferences from a global store using namespaces: `("users", user_id)`.
2. Update/upsert preferences inside a node.
3. Compile the graph passing BOTH a checkpointer and the store.
4. Verify the state is shared across completely separate thread IDs.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


class State(TypedDict):
    user_id: str
    user_input: str
    response: str


# 1. Instantiate the global store (e.g. InMemoryStore)
# In production, this can be a durable store (e.g. PostgresStore / RedisStore)
global_store = InMemoryStore()


# 2. Define node function that interacts with the store
def profile_node(state: State, *, store) -> dict:
    user_id = state["user_id"]
    
    # Define namespace for this user's data
    namespace = ("users", user_id)
    
    # Retrieve current profile from store
    stored_item = store.get(namespace, key="profile")
    profile = stored_item.value if stored_item else {}
    
    print(f"\n[Store Read] Namespace: {namespace}. Profile value: {profile}")
    
    user_text = state["user_input"].lower()
    
    # Detect and save name preference
    if "my name is " in user_text:
        name = state["user_input"].split("my name is ")[-1].strip(".")
        profile["name"] = name
        # Save to store
        store.put(namespace, key="profile", value=profile)
        print(f"[Store Write] Saved name: {name}")
        reply = f"Nice to meet you, {name}!"
    elif profile.get("name"):
        reply = f"Hello {profile['name']}! You asked: '{state['user_input']}'"
    else:
        reply = f"Hello! What is your name?"
        
    return {"response": reply}


# 3. Build and compile graph
builder = StateGraph(State)
builder.add_node("profile_handler", profile_node)
builder.add_edge(START, "profile_handler")
builder.add_edge("profile_handler", END)

# Compile must receive the checkpointer AND the store
graph = builder.compile(
    checkpointer=MemorySaver(),
    store=global_store
)

print("===============================================================================")
print("         LANGGRAPH STATE MANAGEMENT: BASESTORE & INMEMORYSTORE                 ")
print("===============================================================================\n")

# --------------------------------------------------------------------------
# Verify Cross-Thread Persistence
# --------------------------------------------------------------------------
# User ID is 'user_alice'
user_info = {"user_id": "user_alice", "user_input": "", "response": ""}

print("💬 [Thread 1 (session-1)] Store user name preference:")
config_thread_1 = {"configurable": {"thread_id": "session-1"}}
res1 = graph.invoke(
    {**user_info, "user_input": "Hello! My name is Alice."},
    config=config_thread_1
)
print(f"   🤖 Agent Response: {res1['response']}")

print("\n💬 [Thread 2 (session-2)] Access profile on a NEW thread ID:")
# Thread 2 is brand new, but we keep 'user_id' as 'user_alice'
config_thread_2 = {"configurable": {"thread_id": "session-2"}}
res2 = graph.invoke(
    {**user_info, "user_input": "Can you check my order?"},
    config=config_thread_2
)
# The agent should know our name is Alice because the store is shared cross-thread!
print(f"   🤖 Agent Response: {res2['response']}\n")

