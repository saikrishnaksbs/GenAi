"""
Shared vs. Separate State Schemas Between Parent and Subgraph
==================================================================
When a subgraph's state schema shares key names with the parent (as in
the previous file), LangGraph wires them together automatically. But
often a subgraph needs its OWN internal keys that the parent shouldn't
see or manage. In that case, wrap the subgraph invocation in a small
adapter node: a plain function that translates parent state into the
subgraph's input shape, invokes the subgraph explicitly, and translates
its output back. This keeps the subgraph's internal schema fully
encapsulated.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# --- Subgraph with its own private schema, unrelated key names ---
class TranslationState(TypedDict):
    source_text: str
    target_language: str
    translated_text: str
    _internal_token_count: int  # Purely internal bookkeeping, not shared with parent.


def translate_node(state: TranslationState) -> dict:
    return {
        "translated_text": f"[{state['target_language']}] {state['source_text']}",
        "_internal_token_count": len(state["source_text"].split()),
    }


translation_builder = StateGraph(TranslationState)
translation_builder.add_node("translate", translate_node)
translation_builder.add_edge(START, "translate")
translation_builder.add_edge("translate", END)
translation_subgraph = translation_builder.compile()


# --- Parent graph with a completely different schema ---
class DocumentState(TypedDict):
    document: str
    language_pref: str
    localized_document: str


def localize_node(state: DocumentState) -> dict:
    # Adapter: manually build the subgraph's input from parent state...
    sub_input = {
        "source_text": state["document"],
        "target_language": state["language_pref"],
        "translated_text": "",
        "_internal_token_count": 0,
    }
    # ...invoke it explicitly (not wired as a node)...
    sub_result = translation_subgraph.invoke(sub_input)
    # ...and translate only the relevant output back into parent state,
    # discarding the subgraph's private `_internal_token_count`.
    return {"localized_document": sub_result["translated_text"]}


parent_builder = StateGraph(DocumentState)
parent_builder.add_node("localize", localize_node)
parent_builder.add_edge(START, "localize")
parent_builder.add_edge("localize", END)
parent_graph = parent_builder.compile()

result = parent_graph.invoke({
    "document": "Welcome to LangGraph",
    "language_pref": "fr",
    "localized_document": "",
})
print(result)
# -> {"document": "Welcome to LangGraph", "language_pref": "fr",
#     "localized_document": "[fr] Welcome to LangGraph"}
# Note: "_internal_token_count" never leaks into DocumentState.
