"""
CHAT MESSAGE HISTORY STORES
==============================
BaseChatMessageHistory is the abstract interface LangChain uses to persist
conversation messages. RunnableWithMessageHistory (see file 04) is store-
agnostic: it just needs a function that returns some BaseChatMessageHistory
implementation for a given session_id. Different backends let you trade off
between "quick and in-process" and "durable and shared across servers".

This file shows three common implementations: pure in-memory (dev/testing),
Redis-backed (fast, ephemeral-ish, shared across processes), and
Postgres-backed (durable, queryable, survives restarts).
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. In-memory store: fine for local dev, lost on process restart ---
in_memory_history = InMemoryChatMessageHistory()
in_memory_history.add_user_message("Hello!")
in_memory_history.add_ai_message("Hi there, how can I help?")

print(in_memory_history.messages)
# -> [HumanMessage(content="Hello!"), AIMessage(content="Hi there, how can I help?")]


# --- 2. Redis-backed store: shared across app instances, fast, TTL-capable ---
# Requires: pip install langchain-redis redis
from langchain_redis import RedisChatMessageHistory
import redis

print("\n--- 2. Testing Redis Message History Store ---")
try:
    redis_history = RedisChatMessageHistory(
        session_id="user-123",
        redis_url="redis://localhost:6379/0",
        ttl=3600,  # optional: auto-expire the conversation after 1 hour of inactivity
    )
    redis_history.add_message(HumanMessage(content="What's the weather like?"))
    redis_history.add_message(AIMessage(content="I don't have live weather access."))
    print("Redis messages:", redis_history.messages)
    # Messages are serialized to Redis under a key derived from session_id, so
    # any process pointing at the same Redis instance sees the same history.
except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
    print(f"Skipped/Failed to connect to Redis: {e}")
    print("Note: To run this step, ensure a local Redis server is running at localhost:6379")


# --- 3. Postgres-backed store: durable, survives restarts, queryable via SQL ---
# Requires: pip install langchain-postgres psycopg
from langchain_postgres import PostgresChatMessageHistory
import psycopg

print("\n--- 3. Testing Postgres Message History Store ---")
try:
    sync_connection = psycopg.connect("postgresql://user:password@localhost:5432/chatdb", connect_timeout=2)
    
    # One-time setup: creates the message table if it doesn't already exist
    PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
    
    postgres_history = PostgresChatMessageHistory(
        "chat_history",       # table name
        "user-123",            # session_id
        sync_connection=sync_connection,
    )
    postgres_history.add_messages(
        [HumanMessage(content="Remember that I'm vegetarian.")]
    )
    
    print("Postgres messages:", postgres_history.messages)
    # On the next app restart, this same query returns the same messages, since
    # they're persisted as rows in the "chat_history" Postgres table.
except psycopg.OperationalError as e:
    print(f"Skipped/Failed to connect to Postgres: {e}")
    print("Note: To run this step, ensure a local PostgreSQL instance is running at localhost:5432")

