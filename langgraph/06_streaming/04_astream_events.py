"""
astream_events for Fine-Grained Event Streaming
====================================================
`astream_events` (async) exposes the finest-grained view of graph
execution: a stream of individual events like `on_chain_start`,
`on_chat_model_stream`, `on_tool_start`, `on_tool_end`, etc, tagged with
which node/run produced them. This is the mode to reach for when you
need to distinguish "which specific sub-component" produced a token or
side effect - e.g. showing tool-call progress AND streamed tokens in the
same UI, with clear provenance for each.
"""

import asyncio
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It's sunny in {city}."


llm = ChatOllama(model="qwen2.5:1.5b").bind_tools([get_weather])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def agent_node(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()


async def main():
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="What's the weather in Austin?")]},
        version="v2",  # v2 is the current stable event schema.
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # Fine-grained: individual token chunks from the LLM call inside agent_node.
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif kind == "on_chain_start" and event["name"] == "agent":
            print(f"\n[event] entering node: {event['name']}")
        elif kind == "on_tool_start":
            print(f"\n[event] tool call started: {event['name']} args={event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"[event] tool call finished: {event['data'].get('output')}")


asyncio.run(main())
# -> [event] entering node: agent
# -> [event] tool call started: get_weather args={'city': 'Austin'}
# -> [event] tool call finished: It's sunny in Austin.
# -> (streamed tokens of the final natural-language response)
