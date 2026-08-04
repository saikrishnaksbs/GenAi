"""
MemorySaver: In-Memory Checkpointer
=======================================
A checkpointer persists the graph's state after every super-step, keyed
by a `thread_id`. `MemorySaver` keeps checkpoints in process memory - it
disappears when the process exits, which makes it ideal for local
development, tests, and demos, but unsuitable for production durability
(use SqliteSaver/PostgresSaver for that, see the next file). Passing a
checkpointer to `compile()` is what unlocks multi-turn memory, time
travel, and human-in-the-loop interrupts.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot_node(state: ChatState) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Echo: {last}")]}


builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()
# Compiling with a checkpointer makes the graph "stateful" across invocations
# that share the same thread_id.
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "conversation-1"}}

# First turn: only the new human message needs to be passed in -- the
# checkpointer will merge it with any prior history for this thread_id.
graph.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

# Second turn on the same thread_id: prior messages are loaded automatically.
result = graph.invoke({"messages": [HumanMessage(content="How are you?")]}, config=config)
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
# -> HumanMessage: Hi
# -> AIMessage: Echo: Hi
# -> HumanMessage: How are you?
# -> AIMessage: Echo: How are you?

# A different thread_id starts with completely fresh state.
other_config = {"configurable": {"thread_id": "conversation-2"}}
fresh = graph.invoke({"messages": [HumanMessage(content="New chat")]}, config=other_config)
print(len(fresh["messages"]))
# -> 2 (no history from conversation-1 leaks in)
