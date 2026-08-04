"""
RUNNABLE WITH MESSAGE HISTORY
================================
RunnableWithMessageHistory is the modern (LCEL-based) replacement for the
legacy `langchain.memory` classes. Instead of a stateful Memory object glued
to a Chain, you wrap any Runnable (e.g. a prompt | llm pipeline) and supply a
function that returns a ChatMessageHistory for a given session_id. LangChain
then handles reading history in before the call and appending the new
turn after, keyed per-session.

This decouples "how history is stored" (in-memory dict, Redis, Postgres, etc.)
from "how the chain uses it", and works cleanly with .invoke/.stream/.batch.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),  # prior turns get injected here
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# A simple in-memory store mapping session_id -> ChatMessageHistory.
# Swap this for a Redis- or Postgres-backed store in production (see file 05).
_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",   # which input key holds the new human message
    history_messages_key="history",  # must match the MessagesPlaceholder name
)

# Every call must pass a session_id via `config` so history is scoped correctly.
config = {"configurable": {"session_id": "user-123"}}

response_1 = chain_with_history.invoke({"input": "Hi, I'm Sai."}, config=config)
print(response_1.content)
# -> "Hello Sai! How can I help you today?"

response_2 = chain_with_history.invoke({"input": "What's my name?"}, config=config)
print(response_2.content)
# -> "Your name is Sai."

# A different session_id gets a completely independent history
other_config = {"configurable": {"session_id": "user-456"}}
response_3 = chain_with_history.invoke({"input": "What's my name?"}, config=other_config)
print(response_3.content)
# -> "I don't know your name yet — you haven't told me."

# Inspect the raw stored messages for a session
print(_store["user-123"].messages)
# -> [HumanMessage(content="Hi, I'm Sai."), AIMessage(content="Hello Sai!..."),
#     HumanMessage(content="What's my name?"), AIMessage(content="Your name is Sai.")]
