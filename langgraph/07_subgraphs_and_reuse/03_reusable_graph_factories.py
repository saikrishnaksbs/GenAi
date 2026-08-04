"""
Reusable Graph Components as Functions/Factories
====================================================
To reuse a graph pattern across multiple parent graphs (or with
different configuration), wrap its construction in a factory function
that returns a freshly compiled graph. This avoids sharing mutable
builder state between call sites and lets each caller parameterize the
subgraph (different tools, prompts, or models) without duplicating its
structure.
"""

from typing import TypedDict, Callable
from langgraph.graph import StateGraph, START, END


class ValidationState(TypedDict):
    value: str
    is_valid: bool
    error: str


def make_validation_subgraph(validator_fn: Callable[[str], bool], error_message: str):
    """Factory: builds and compiles a small reusable validation graph.

    Each call produces an independent compiled graph, so the same factory
    can be used to create differently-configured validators (e.g. one for
    emails, one for phone numbers) without them interfering with each other.
    """

    def validate_node(state: ValidationState) -> dict:
        valid = validator_fn(state["value"])
        return {"is_valid": valid, "error": "" if valid else error_message}

    builder = StateGraph(ValidationState)
    builder.add_node("validate", validate_node)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", END)
    return builder.compile()


# Two independently configured, reusable subgraphs from the same factory.
email_validator = make_validation_subgraph(
    validator_fn=lambda v: "@" in v and "." in v,
    error_message="Not a valid email address",
)
phone_validator = make_validation_subgraph(
    validator_fn=lambda v: v.isdigit() and len(v) == 10,
    error_message="Not a valid 10-digit phone number",
)

print(email_validator.invoke({"value": "user@example.com", "is_valid": False, "error": ""}))
# -> {"value": "user@example.com", "is_valid": True, "error": ""}

print(phone_validator.invoke({"value": "12345", "is_valid": False, "error": ""}))
# -> {"value": "12345", "is_valid": False, "error": "Not a valid 10-digit phone number"}

# Because each is its own compiled graph, they can be plugged into
# different parent graphs (e.g. a signup form graph) as ordinary nodes.
