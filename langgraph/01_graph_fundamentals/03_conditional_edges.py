"""
Conditional Edges and Routing Functions
==========================================
A regular edge always goes A -> B. A conditional edge instead runs a
routing function after a node, and the function's return value (a node
name, or list of names) decides where to go next. This is how you build
branches: "if the LLM asked for a tool, go to the tool node; otherwise
end." `add_conditional_edges` takes the source node, the routing
function, and optionally a mapping from routing-function outputs to
actual node names.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    difficulty: str
    answer: str


def classify_node(state: State) -> dict:
    # Pretend classification: long questions are "hard".
    difficulty = "hard" if len(state["question"]) > 40 else "easy"
    return {"difficulty": difficulty}


def easy_answer_node(state: State) -> dict:
    return {"answer": "Quick heuristic answer."}


def hard_answer_node(state: State) -> dict:
    return {"answer": "Deep multi-step reasoning answer."}


def route_by_difficulty(state: State) -> Literal["easy_answer", "hard_answer"]:
    # Routing functions read state and return the name of the next node.
    if state["difficulty"] == "hard":
        return "hard_answer"
    return "easy_answer"


builder = StateGraph(State)
builder.add_node("classify", classify_node)
builder.add_node("easy_answer", easy_answer_node)
builder.add_node("hard_answer", hard_answer_node)

builder.add_edge(START, "classify")
# The mapping dict (3rd arg) is optional here since the function's return
# values already match the real node names, but it's shown for clarity.
builder.add_conditional_edges(
    "classify",
    route_by_difficulty,
    {"easy_answer": "easy_answer", "hard_answer": "hard_answer"},
)
builder.add_edge("easy_answer", END)
builder.add_edge("hard_answer", END)

graph = builder.compile()

print(graph.invoke({"question": "What is 2+2?", "difficulty": "", "answer": ""}))
# -> {..., "difficulty": "easy", "answer": "Quick heuristic answer."}

print(graph.invoke({
    "question": "Explain the trade-offs of eventual consistency in distributed databases",
    "difficulty": "",
    "answer": "",
}))
# -> {..., "difficulty": "hard", "answer": "Deep multi-step reasoning answer."}
