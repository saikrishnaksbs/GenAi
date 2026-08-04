"""
TOOL SCHEMAS WITH PYDANTIC ARGS_SCHEMA
=========================================
For tools with multiple arguments, types that need validation, or fields
that need rich descriptions (which help the model call the tool
correctly), you define an explicit Pydantic model and pass it as
`args_schema`. This gives you field-level descriptions, defaults, and
validation beyond what type hints alone can express.
"""

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field


class WeatherQuery(BaseModel):
    """Input schema for the get_weather tool."""

    city: str = Field(description="the city to look up weather for, e.g. 'Paris'")
    units: str = Field(
        default="celsius",
        description="temperature units, either 'celsius' or 'fahrenheit'",
    )


# args_schema can be attached directly to the @tool decorator.
@tool(args_schema=WeatherQuery)
def get_weather(city: str, units: str = "celsius") -> str:
    """Look up the current weather for a city."""
    # Pretend implementation — in reality this would call a weather API.
    temp = 18 if units == "celsius" else 64
    return f"It is currently {temp} degrees {units} in {city}."


print(get_weather.args)
# -> {
#      "city": {"description": "the city to look up weather for, e.g. 'Paris'", "title": "City", "type": "string"},
#      "units": {"default": "celsius", "description": "temperature units, ...", "title": "Units", "type": "string"},
#    }
print(get_weather.invoke({"city": "Tokyo", "units": "celsius"}))
# -> "It is currently 18 degrees celsius in Tokyo."


# The same pattern works with StructuredTool.from_function for functions
# you don't control the definition of.
class SearchQuery(BaseModel):
    query: str = Field(description="the search query string")
    max_results: int = Field(default=5, description="maximum number of results to return")


def web_search(query: str, max_results: int = 5) -> list[str]:
    # Placeholder implementation.
    return [f"result {i} for '{query}'" for i in range(1, max_results + 1)]


search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the web and return a list of result snippets.",
    args_schema=SearchQuery,
)

print(search_tool.invoke({"query": "LangChain tools", "max_results": 2}))
# -> ["result 1 for 'LangChain tools'", "result 2 for 'LangChain tools'"]

# A well-defined args_schema is what lets the model's function-calling API
# produce correctly typed, validated arguments — the schema is serialized
# to JSON Schema and sent as part of the tool-calling request.
print(search_tool.args_schema.model_json_schema())
# -> {"title": "SearchQuery", "type": "object", "properties": {...}, "required": ["query"]}
