"""
LANGGRAPH AGENT (Modern Replacement for AgentExecutor)
======================================================
In modern LangChain development, AgentExecutor is deprecated in favor of LangGraph.
LangGraph compiles your agent flow as a state graph containing nodes and edges:
- Nodes represent computation (calling LLM, calling tools).
- Edges represent transitions (conditional check to see if more tools are needed).

`create_react_agent` from `langgraph.prebuilt` is the standard pre-built graph
that replaces AgentExecutor, matching its API while providing superior flexibility.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# Define tools for the agent
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# Initialize tools and model
tools = [add, multiply]
model = ChatOllama(model="qwen2.5:1.5b")

# Compile the agent using LangGraph's prebuilt react agent.
# Use state_modifier to provide system prompts.



# agent_graph = create_react_agent(
#     model,
#     tools,
#     state_modifier="You are a math assistant. Use tools for all calculations."
# )



def custom_modifier(state: dict):
    """
    This function is automatically called by LangGraph 
    right before sending the conversation history to the LLM.
    """
    print("\n--- [DEBUG] custom_modifier called by LangGraph! ---")
    print(f"State type: {type(state)}")
    print(f"State keys: {list(state.keys())}")
    print(f"Current message count: {len(state['messages'])}")
    
    # 1. Retrieve the existing messages from graph state
    messages = state["messages"]
    
    # 2. Dynamically define system instruction (e.g., force a specific persona/behavior)
    dynamic_instruction = SystemMessage(
        content="You are a poetic math assistant. Explain all math steps in rhyming sentences."
    )
    
    # 3. Return the system instruction prepended to the messages list
    return [dynamic_instruction] + messages
# ==========================================
# 3. COMPILE AND RUN THE GRAPH
# ==========================================
# We pass the custom modifier function to state_modifier
agent_graph = create_react_agent(
    model,
    tools,
    state_modifier=custom_modifier
)


# Run the agent and stream events
print("=== Running LangGraph Agent ===")
input_data = {"messages": [("user", "What is (3 + 4) multiplied by 5?")]}

# We iterate over step stream updates
for event in agent_graph.stream(input_data, stream_mode="values"):
    # Print messages as they are added to the graph state
    for message in event.get("messages", []):
        message.pretty_print()
