"""
Defining State with Pydantic BaseModel
==========================================
Instead of a TypedDict, a LangGraph state schema can be a Pydantic
`BaseModel`. LangGraph will validate and coerce the state on every node
boundary using Pydantic's rules, and raise a `ValidationError` if a node
returns data that doesn't fit the schema. This trades a small perf cost
for much stronger guarantees - useful when state is complex, comes from
untrusted input, or you want field validators/defaults enforced
automatically.
"""

from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END


class SupportTicketState(BaseModel):
    subject: str
    priority: int = Field(default=3, ge=1, le=5)  # Constrained to 1-5.
    resolved: bool = False

    @field_validator("subject")
    @classmethod
    def subject_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("subject must not be empty")
        return v


def triage_node(state: SupportTicketState) -> dict:
    # Access fields as attributes, not dict keys, when state is Pydantic.
    urgent_keywords = ("down", "urgent", "outage")
    priority = 1 if any(k in state.subject.lower() for k in urgent_keywords) else 3
    return {"priority": priority}


def resolve_node(state: SupportTicketState) -> dict:
    return {"resolved": state.priority >= 4}  # Pretend low-priority tickets auto-resolve.


builder = StateGraph(SupportTicketState)
builder.add_node("triage", triage_node)
builder.add_node("resolve", resolve_node)
builder.add_edge(START, "triage")
builder.add_edge("triage", "resolve")
builder.add_edge("resolve", END)
graph = builder.compile()

# Input can be passed as a dict or a model instance; LangGraph validates it.
result = graph.invoke(SupportTicketState(subject="Production database is down"))
print(result)
# -> {"subject": "Production database is down", "priority": 1, "resolved": False}

# Invalid input raises a validation error before any node runs.
try:
    graph.invoke({"subject": "", "priority": 3, "resolved": False})
except Exception as exc:
    print(type(exc).__name__)
    # -> ValidationError (or wrapped equivalent, depending on LangGraph version)
