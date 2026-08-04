"""
Exposing a Graph via the LangGraph SDK Client
=================================================
Once a graph is deployed to LangGraph Platform (or served locally via
`langgraph dev`), it's reachable over a REST API. Rather than hand-rolling
HTTP calls, the `langgraph_sdk` package provides a typed Python client
(`get_client`, `get_sync_client`) for creating threads, starting runs,
and streaming results, mirroring the LangServe-style API the platform
exposes under the hood. This file demonstrates the client side -- code
that would run in a separate consumer application talking to the
deployed graph, not the graph definition itself.
"""

import asyncio
from langgraph_sdk import get_client


async def main():
    # Connects to a locally running `langgraph dev` server or a deployed
    # LangGraph Platform URL.
    client = get_client(url="http://localhost:2024")

    # Threads are the SDK's equivalent of a checkpointer's thread_id --
    # they group runs into a persistent, resumable conversation.
    thread = await client.threads.create()
    print(thread["thread_id"])
    # -> "b1a7e3c2-...."

    # `assistants.search` lists deployed graphs (by the names declared in
    # langgraph.json's "graphs" mapping, e.g. "support_agent").
    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]

    # Stream a run over HTTP exactly like a local graph.stream() call --
    # same stream_mode options (values/updates/messages) apply here too.
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "What's my order status?"}]},
        stream_mode="updates",
    ):
        print(chunk.event, chunk.data)
    # -> metadata {"run_id": "..."}
    # -> updates {"support": {"messages": [...]}}
    # -> end None

    # Later, resume the same conversation by reusing thread["thread_id"] --
    # the platform's persistence layer plays the same role MemorySaver/
    # SqliteSaver play locally.
    history = await client.threads.get_history(thread["thread_id"])
    print(len(history))
    # -> number of checkpoints recorded for this thread so far


asyncio.run(main())

# A synchronous alternative exists for non-async codebases:
#   from langgraph_sdk import get_sync_client
#   client = get_sync_client(url="http://localhost:2024")
