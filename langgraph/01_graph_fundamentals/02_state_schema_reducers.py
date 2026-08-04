"""
State Schema Design: TypedDict, Pydantic, and Reducers
========================================================
A graph's State schema defines the shape of data flowing between nodes.
LangGraph accepts either a `TypedDict` (lightweight, no runtime validation)
or a Pydantic `BaseModel` (runtime validation, type coercion). By default,
each key is *overwritten* by whatever a node returns. Wrapping a field's
type in `Annotated[type, reducer_fn]` changes that: instead of overwriting,
LangGraph calls the reducer with (current_value, new_value) to combine
them. `operator.add` is the classic reducer for accumulating lists, and
`add_messages` is a specialized reducer for chat message history.
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


# --- Option A: TypedDict state with reducers ---
class State(TypedDict):
    # Default behavior: last write wins (no Annotated wrapper).
    topic: str
    # operator.add concatenates lists returned by nodes instead of replacing.
    notes: Annotated[list[str], operator.add]
    # add_messages appends new messages and de-dupes/updates by message id.
    messages: Annotated[list[AnyMessage], add_messages]


def add_note_a(state: State) -> dict:
    return {"notes": ["first observation"]}


def add_note_b(state: State) -> dict:
    return {"notes": ["second observation"]}


builder = StateGraph(State)
builder.add_node("note_a", add_note_a)
builder.add_node("note_b", add_note_b)
builder.add_edge(START, "note_a")
builder.add_edge("note_a", "note_b")
builder.add_edge("note_b", END)
graph = builder.compile()

result = graph.invoke({"topic": "demo", "notes": [], "messages": [HumanMessage("hi")]})
print(result["notes"])
# -> ["first observation", "second observation"]  (accumulated, not overwritten)


# --- Option B: Pydantic BaseModel state ---
from pydantic import BaseModel, Field


class PydanticState(BaseModel):
    topic: str
    # Pydantic fields can also use Annotated reducers.
    notes: Annotated[list[str], operator.add] = Field(default_factory=list)


def add_note_p(state: PydanticState) -> dict:
    # Pydantic state gives runtime validation: invalid types raise here.
    return {"notes": [f"validated note about {state.topic}"]}


p_builder = StateGraph(PydanticState)
p_builder.add_node("add_note", add_note_p)
p_builder.add_edge(START, "add_note")
p_builder.add_edge("add_note", END)
p_graph = p_builder.compile()

p_result = p_graph.invoke(PydanticState(topic="pydantic demo"))
print(p_result)
# -> {"topic": "pydantic demo", "notes": ["validated note about pydantic demo"]}


# --- Custom reducer function ---
def keep_longest(current: str, new: str) -> str:
    # Custom reducers can implement arbitrary merge logic.
    return new if len(new) > len(current) else current


class CustomReducerState(TypedDict):
    best_answer: Annotated[str, keep_longest]
