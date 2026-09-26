"""
Cycles and Loops
===================
Plain LCEL chains are directed acyclic graphs: data flows forward once.
Agents, however, often need to loop — call a tool, check the result, call
another tool, retry on failure — until some condition is met. LangGraph
allows edges that point *backward*, forming a cycle. Combined with a
conditional edge that decides "loop again" vs. "exit", this is the
foundation of the ReAct agent pattern. Here we model a simple retry loop
that keeps attempting a flaky operation until it succeeds or a max
attempt count is hit.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    attempts: int
    max_attempts: int
    success: bool
    result: str


def attempt_operation(state: State) -> dict:
    attempts = state["attempts"] + 1
    # Simulate success only on the 3rd attempt.
    success = attempts >= 3
    return {
        "attempts": attempts,
        "success": success,
        "result": "done" if success else "failed",
    }


def should_retry(state: State) -> Literal["retry", "give_up", "done"]:
    if state["success"]:
        return "done"
    if state["attempts"] >= state["max_attempts"]:
        return "give_up"
    return "retry"  # Loop back to attempt again.


builder = StateGraph(State)
builder.add_node("attempt", attempt_operation)

builder.add_edge(START, "attempt")
builder.add_conditional_edges(
    "attempt",
    should_retry,
    {
        "retry": "attempt",  # Cycle: this edge points back to the same node.
        "give_up": END,
        "done": END,
    },
)

print("===============================================================================")
print("                LANGGRAPH FUNDAMENTALS: CYCLES & RETRY LOOPS                   ")
print("===============================================================================\n")

graph = builder.compile()

print("🔄 [Scenario 1] Standard Retry Loop (max_attempts = 5):")
result = graph.invoke({"attempts": 0, "max_attempts": 5, "success": False, "result": ""})
print(f"   • Attempts Taken : {result['attempts']}")
print(f"   • Success Status : {result['success']}")
print(f"   • Final Result   : {result['result']}\n")

print("🛑 [Scenario 2] Max Attempts Exceeded Loop (max_attempts = 2):")
give_up_result = graph.invoke({"attempts": 0, "max_attempts": 2, "success": False, "result": ""})
print(f"   • Attempts Taken : {give_up_result['attempts']}")
print(f"   • Success Status : {give_up_result['success']}")
print(f"   • Final Result   : {give_up_result['result']}\n")

