# LangGraph State Management

This directory details how data flows and persists inside a LangGraph application. It covers state schema designs (TypedDict vs. Pydantic), custom and built-in reducers, message tracking with identity-aware merges, and decoupled partial state updates.

---

## Table of Contents
1. [State Definition: TypedDict vs. Pydantic](#1-state-definition-typeddict-vs-pydantic)
2. [Reducers: Accumulators & Custom Merging](#2-reducers-accumulator--custom-merging)
3. [The add_messages Reducer (Message Identity Merges)](#3-the-add_messages-reducer-message-identity-merges)
4. [Partial State Updates & Decoupled Channels](#4-partial-state-updates--decoupled-channels)
5. [BaseStore & InMemoryStore (Cross-Thread Long-Term Memory)](#5-basestore--inmemorystore-cross-thread-long-term-memory)

---

## 1. State Definition: TypedDict vs. Pydantic

### TypedDict State
[TypedDict](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/01_typeddict_state.py#L4-L12) is the standard and most lightweight way to define a graph schema. It provides static typing and autocomplete support in IDEs without runtime validation overhead. Nodes read from the dictionary and return partial dictionary updates.

### Pydantic State
Using a [Pydantic BaseModel](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/02_pydantic_state.py#L4-L14) enforces runtime validation and type coercion.
- LangChain validates the dictionary returned by a node at each boundary.
- If a value does not match the schema constraints, Pydantic raises a `ValidationError`.
- Recommended when processing external, untrusted user inputs or when enforcing strict constraints (e.g. integer bounds, regex string checks).

---

## 2. Reducers: Accumulators & Custom Merging
By default, if two nodes write to the same state key, the second node **overwrites** the value. To change this, you annotate the field with a **reducer function**: `Annotated[Type, reducer_fn]`.
- A reducer is a function that accepts `(current_value, new_value)` and returns a merged value.
- **`operator.add`** is the standard reducer to append items to lists or sum numbers (covered in [03_reducers.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/03_reducers.py#L4-L12)).
- You can write custom reducer functions to merge dictionaries, manage sets, or maintain sliding window histories.

```python
from typing import Annotated, TypedDict
import operator

class GraphState(TypedDict):
    # Values will be appended to list, not overwritten
    history: Annotated[list[str], operator.add]
```

---

## 3. The add_messages Reducer (Message Identity Merges)
Chat interactions rely on lists of messages. Appending messages using `operator.add` can lead to duplicate entries, especially when regenerating or streaming messages.
To resolve this, LangGraph provides the **`add_messages`** reducer (covered in [04_add_messages_reducer.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/04_add_messages_reducer.py)):
- It appends new messages to the message list.
- **Identity Matching**: If a message in the new list has the same `id` as an existing message in the state, the existing message is **replaced in-place**.
- This is key for streaming token chunks or modifying message arguments dynamically during execution.

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # Handles appends and replacements automatically
    messages: Annotated[list[AnyMessage], add_messages]
```

---

## 4. Partial State Updates & Decoupled Channels
Nodes do not need to return the entire state schema (details in [05_partial_state_updates.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/05_partial_state_updates.py)).
- A node only needs to return a dictionary with the keys it wants to update (referred to as **channels**).
- LangGraph merges these partial updates into the global state using each key's registered reducer (falling back to overwriting if no reducer is defined).
- This decoupling allows you to design large state schemas without tightly coupling individual node functions to unrelated keys.

---

## 5. BaseStore & InMemoryStore (Cross-Thread Long-Term Memory)
While Checkpointers manage the state within a single thread (`thread_id`), some applications require persistent memories that span across different conversations and users.
- **`BaseStore`**: The standard interface for storing arbitrary key-value pairs categorized by namespace tuples (e.g. `("users", user_id)`).
- **`InMemoryStore`**: An in-memory implementation of the store, useful for development. Other options include `PostgresStore` and `RedisStore`.
- **Node Access**: Nodes can request the `store` argument automatically during execution by defining it in the function signature.

See [06_base_store_memory.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/02_state_management/06_base_store_memory.py) for details.
