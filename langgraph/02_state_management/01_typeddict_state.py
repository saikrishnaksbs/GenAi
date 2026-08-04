"""
Defining State with TypedDict
=================================
`TypedDict` is the most common way to define a LangGraph state schema.
It's a plain dictionary shape with type hints - no runtime validation,
minimal overhead, and IDE autocomplete for keys. Nodes receive the state
as a regular dict and return a dict of the fields they want to update.
This is the right choice when you trust your own node code and don't
need Pydantic's validation/coercion at graph boundaries.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END


class OrderState(TypedDict):
    order_id: str
    item: str
    quantity: int
    # Optional fields can be absent until a node fills them in.
    shipping_estimate: Optional[str]


def price_check_node(state: OrderState) -> dict:
    # Access fields with normal dict indexing.
    print(f"Checking price for {state['quantity']}x {state['item']}")
    return {}  # No state change needed here.


def shipping_node(state: OrderState) -> dict:
    days = 2 if state["quantity"] < 10 else 5
    return {"shipping_estimate": f"{days} business days"}


builder = StateGraph(OrderState)
builder.add_node("price_check", price_check_node)
builder.add_node("shipping", shipping_node)
builder.add_edge(START, "price_check")
builder.add_edge("price_check", "shipping")
builder.add_edge("shipping", END)
graph = builder.compile()

result = graph.invoke({
    "order_id": "A123",
    "item": "widget",
    "quantity": 3,
    "shipping_estimate": None,
})
print(result)
# -> {"order_id": "A123", "item": "widget", "quantity": 3, "shipping_estimate": "2 business days"}
