"""
thread_id and Multi-Turn Conversations Across Invocations
==============================================================
`thread_id` is the key that ties a sequence of `.invoke()` calls into one
continuous conversation. Every invocation with the same `thread_id`
resumes from that thread's latest checkpoint instead of starting over -
you don't need to manually re-pass the full message history each time.
This is the mechanism that turns a stateless function call into a
stateful, resumable session.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_count: int


def chatbot_node(state: ChatState) -> dict:
    turn = state.get("turn_count", 0) + 1
    last = state["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"(turn {turn}) You said: {last}")],
        "turn_count": turn,
    }


builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=MemorySaver())


def send(thread_id: str, text: str):
    # Only the NEW message is sent -- LangGraph loads prior state for this
    # thread_id from the checkpointer and merges it via the add_messages reducer.
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    print(f"[{thread_id}] {result['messages'][-1].content}")
    return result


send("alice", "Hi, I'm Alice")
send("alice", "What's my name?")
# -> [alice] (turn 1) You said: Hi, I'm Alice
# -> [alice] (turn 2) You said: What's my name?

# A separate thread_id is a completely independent conversation, with its
# own turn_count and message history -- concurrent users never collide.
send("bob", "Hi, I'm Bob")
# -> [bob] (turn 1) You said: Hi, I'm Bob

# Inspect the persisted state for a thread directly via get_state().
state_snapshot = graph.get_state({"configurable": {"thread_id": "alice"}})
print(state_snapshot.values["turn_count"])
# -> 2
