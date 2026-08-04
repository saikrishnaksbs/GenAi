# LangChain Conversation Memory & History

This directory details LangChain's abstractions for maintaining conversation context across multi-turn interactions. It covers legacy stateful memory classes, token-limited buffers, LLM-based summary folding, the modern LCEL-compatible wrapper, and database-backed persistent message history stores.

---

## Table of Contents
1. [Legacy memory classes](#1-legacy-memory-classes)
2. [Conversation Summary & Hybrid Memory](#2-conversation-summary--hybrid-memory)
3. [Token-Budget Memory](#3-token-budget-memory)
4. [Modern LCEL Memory: RunnableWithMessageHistory](#4-modern-lcel-memory-runnablewithmessagehistory)
5. [Message History Persistence Backends](#5-message-history-persistence-backends)

---

## 1. Legacy Memory Classes
Historically, LangChain relied on stateful wrappers that managed messages internally and formatted them into prompt inputs (details in [01_conversation_buffer_memory.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/07_memory/01_conversation_buffer_memory.py)):
- **`ConversationBufferMemory`**: Stores all conversation turns verbatim. The prompt grows unbounded, which will eventually exceed the model's context length limit.
- **`ConversationBufferWindowMemory`**: Restricts the stored history to the last `k` turns, keeping prompt sizes stable but discarding older history entirely.

---

## 2. Conversation Summary & Hybrid Memory
To manage history without exceeding context limits, LangChain supports summarization-based memory classes (detailed in [02_conversation_summary_memory.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/07_memory/02_conversation_summary_memory.py)):
- **`ConversationSummaryMemory`**: Calls an LLM on every turn to summarize the conversation history, resulting in a constant-sized prompt segment.
- **`ConversationSummaryBufferMemory`**: A hybrid approach that keeps recent messages verbatim up to a token limit, and summarizes older turns once that limit is reached.

---

## 3. Token-Budget Memory
[ConversationTokenBufferMemory](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/07_memory/03_conversation_token_buffer_memory.py#L4-L12) is a token-aware window buffer.
- Instead of using a fixed message count (like window memory), it monitors the token count using the model's tokenizer.
- When new messages exceed the `max_token_limit`, the memory drops the oldest turns one by one until the total token size is within the budget.

---

## 4. Modern LCEL Memory: RunnableWithMessageHistory
The modern approach in LangChain replaces stateful memory objects with [RunnableWithMessageHistory](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/07_memory/04_runnable_with_message_history.py#L4-L10).
- This wrapper decorates an existing LCEL chain.
- It accepts a `get_session_history` factory function that returns a storage backend instance for a given `session_id`.
- The wrapper reads the message history before invocation, formats it into a matching `MessagesPlaceholder` template slot, and writes the model's new response back to the storage database after execution.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the prompt | model chain
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Invoke with a session_id inside the config
with_history.invoke(
    {"input": "Hi!"},
    config={"configurable": {"session_id": "session_123"}}
)
```

---

## 5. Message History Persistence Backends
The `get_session_history` function returns an implementation of `BaseChatMessageHistory`. In [05_chat_message_history_stores.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/07_memory/05_chat_message_history_stores.py), three common storage backends are demonstrated:
- **`ChatMessageHistory`**: Pure in-memory dictionary storage (ideal for local testing and dev tasks).
- **`RedisChatMessageHistory`**: Stored in a Redis cache (ideal for fast, multi-process cloud environments with key expirations).
- **`PostgresChatMessageHistory`**: Stored in a relational database (ideal for durable long-term storage and structured relational querying).
