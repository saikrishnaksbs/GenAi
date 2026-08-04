"""
TOOL DECORATOR AND STRUCTURED TOOL
=====================================
LangChain "tools" are functions a model can choose to call. There are three
common ways to define one:

  - `@tool` decorator: turns a plain Python function into a Tool, inferring
    name/args/description from the function signature and docstring.
  - `StructuredTool.from_function`: same idea but more explicit, useful when
    wrapping an existing function you don't want to decorate directly.
  - Subclassing `BaseTool`: full control, needed for complex tools with
    custom run/arun behavior or extra configuration.
"""

from langchain_core.tools import BaseTool, StructuredTool, tool


# --- @tool decorator -----------------------------------------------------
@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    # The docstring becomes the tool's description shown to the model.
    return len(word)


print(get_word_length.name)
# -> "get_word_length"
print(get_word_length.description)
# -> "Return the number of characters in a word."
print(get_word_length.args)
# -> {"word": {"title": "Word", "type": "string"}}
print(get_word_length.invoke({"word": "langchain"}))
# -> 9


# --- StructuredTool.from_function -----------------------------------------
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two integers and return the product.",
)

print(multiply_tool.invoke({"a": 6, "b": 7}))
# -> 42


# --- Subclassing BaseTool --------------------------------------------------
class ReverseStringTool(BaseTool):
    """A tool defined as a class, useful when you need custom init logic
    or state that a plain function/decorator can't easily hold."""

    name: str = "reverse_string"
    description: str = "Reverse the characters in the given string."

    def _run(self, text: str) -> str:
        # Synchronous execution path.
        return text[::-1]

    async def _arun(self, text: str) -> str:
        # Async execution path; falls back to _run if omitted in newer
        # versions, but defining it explicitly avoids the default warning.
        return self._run(text)


reverse_tool = ReverseStringTool()
print(reverse_tool.invoke({"text": "LangChain"}))
# -> "niahCgnaL"

# All three tool styles are interchangeable — a model-binding call like
# `model.bind_tools([...])` accepts any mix of them.
all_tools = [get_word_length, multiply_tool, reverse_tool]
print([t.name for t in all_tools])
# -> ["get_word_length", "multiply", "reverse_string"]
