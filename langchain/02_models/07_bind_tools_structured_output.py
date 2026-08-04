"""
BIND_TOOLS() AND WITH_STRUCTURED_OUTPUT()
============================================
Two related but distinct ways to get structured behavior from a chat model:

    .bind_tools(tools)          -> the model may choose to call one or more
                                    tools instead of (or alongside) replying
                                    in plain text. You get an AIMessage with
                                    a `tool_calls` list back.
    .with_structured_output(schema) -> forces the model's entire response to
                                    conform to a schema, returning a parsed
                                    Python object (e.g. a Pydantic model)
                                    directly instead of an AIMessage.

Use bind_tools when the model needs to decide *whether* to act. Use
with_structured_output when you always want a specific shaped answer.
"""

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)


# --- bind_tools(): model optionally calls tools -------------------------
@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a given city."""
    return f"It's sunny in {city}."


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Kyoto?")
print(response.tool_calls)
# -> [{'name': 'get_weather', 'args': {'city': 'Kyoto'}, 'id': 'call_xyz789'}]

# If the input doesn't require a tool, the model just replies normally.
response = model_with_tools.invoke("What's 2 + 2?")
print(response.content, response.tool_calls)
# -> "2 + 2 is 4." []


# --- with_structured_output(): force a specific schema -------------------
class MovieReview(BaseModel):
    """A structured summary of a movie review."""

    title: str = Field(description="The movie's title")
    rating: int = Field(description="Rating out of 10")
    sentiment: str = Field(description="One of: positive, negative, mixed")


structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke(
    "Review: 'Dune: Part Two' is a visually stunning epic, though a bit "
    "long. I'd give it an 8 out of 10."
)
print(result)
# -> MovieReview(title='Dune: Part Two', rating=8, sentiment='positive')
print(type(result))
# -> <class '__main__.MovieReview'>

# with_structured_output can also target a raw JSON schema dict, or set
# `method="json_mode"` for providers that support strict JSON-mode decoding.
