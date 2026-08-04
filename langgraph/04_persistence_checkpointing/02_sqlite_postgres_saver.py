"""
SqliteSaver / PostgresSaver for Durable Persistence
=======================================================
For anything beyond local experimentation, checkpoints need to survive
process restarts. `SqliteSaver` writes checkpoints to a SQLite file (or
in-memory DB), and `PostgresSaver` writes to a Postgres database -
both implement the same checkpointer interface as `MemorySaver`, so
swapping between them requires no changes to graph logic. Both ship
with a `.setup()` step that creates the required tables, and both offer
context-manager constructors (`from_conn_string`) that handle the
underlying connection's lifecycle.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# SqliteSaver ships in the `langgraph-checkpoint-sqlite` package.
from langgraph.checkpoint.sqlite import SqliteSaver

# PostgresSaver ships in the `langgraph-checkpoint-postgres` package.
from langgraph.checkpoint.postgres import PostgresSaver


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot_node(state: ChatState) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Ack: {last}")]}


builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# --- SQLite: file-backed, durable across process restarts ---
with SqliteSaver.from_conn_string("checkpoints.sqlite") as sqlite_checkpointer:
    sqlite_checkpointer.setup()  # Creates tables on first run; no-op after.
    sqlite_graph = builder.compile(checkpointer=sqlite_checkpointer)

    config = {"configurable": {"thread_id": "user-42"}}
    sqlite_graph.invoke({"messages": [HumanMessage(content="Remember me")]}, config=config)
    # Even after this `with` block exits and the process restarts, a new
    # SqliteSaver pointed at the same file can resume thread "user-42".

# --- Postgres: production-grade, shared across multiple app instances ---
DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"
with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
    postgres_checkpointer.setup()  # Idempotent schema migration.
    postgres_graph = builder.compile(checkpointer=postgres_checkpointer)

    pg_config = {"configurable": {"thread_id": "user-42"}}
    result = postgres_graph.invoke(
        {"messages": [HumanMessage(content="Hello from prod")]}, config=pg_config
    )
    print(result["messages"][-1].content)
    # -> "Ack: Hello from prod"
