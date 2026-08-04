"""
Streaming LLM Tokens from Inside a Node: stream_mode="messages"
====================================================================
When a node internally calls a chat model, you often want to stream the
model's tokens to the client as they're generated, not wait for the
whole node to finish. `stream_mode="messages"` surfaces every LLM token
(as `AIMessageChunk`) emitted by any chat model invoked inside a node,
tagged with metadata about which node/graph produced it. This only works
for models invoked with LangChain's standard `.stream()`/astream
machinery under the hood.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatOllama


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatOllama(model="qwen2.5:1.5b")


def chatbot_node(state: State) -> dict:
    # Because this node invokes a chat model, LangGraph can tap into that
    # model's token stream when the graph itself is run with stream_mode="messages".
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

for message_chunk, metadata in graph.stream(
    {"messages": [HumanMessage(content="Write a haiku about graphs")]},
    stream_mode="messages",
):
    # `message_chunk` is an AIMessageChunk with a small piece of the response.
    # `metadata` tells you which node ("chatbot") and langgraph_step produced it.
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
# -> streams token-by-token, e.g.: "Nodes branch and loop\nState flows through
#     each vertex now\nGraphs think, step by step"

print()
print("---")
# metadata example for a chunk:
# {"langgraph_node": "chatbot", "langgraph_step": 1, "ls_model_name": "claude-sonnet-4-5", ...}
