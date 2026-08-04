# LangChain Tools & Agentic Capabilities

This directory covers LangChain **Tools**—interfaces that allow models to interact with external systems. It details tool definition patterns, validation schemas, built-in toolkits, execution flows, and advanced control parameters like argument injection and split content/artifact outputs.

---

## Table of Contents
1. [Defining Tools: Decorator, Factory, & Subclassing](#1-defining-tools-decorator-factory--subclassing)
2. [Input Schema Validation with Pydantic](#2-input-schema-validation-with-pydantic)
3. [Built-In Toolkits](#3-built-in-toolkits)
4. [Tool Calling Protocol & Formatting](#4-tool-calling-protocol--formatting)
5. [Error Handling & ToolNode Execution Loop](#5-error-handling--toolnode-execution-loop)
6. [Advanced: Injected Tools Arguments & Split Artifacts](#6-advanced-injected-tools-arguments--split-artifacts)
7. [Sandboxing Tool Execution & Least-Privilege Tools](#7-sandboxing-tool-execution--least-privilege-tools)

---

## 1. Defining Tools: Decorator, Factory, & Subclassing
A Tool represents a function the LLM can call. LangChain supports three main definitions:
- **`@tool` decorator**: Automatically creates a tool from a function, inferring the name, arguments, types, and using the docstring as the description.
- **`StructuredTool.from_function`**: Utility function to wrap pre-existing functions without decorating them directly.
- **Subclassing `BaseTool`**: Inheriting from the class gives full control, allowing custom states or overrides for synchronous (`_run`) and asynchronous (`_arun`) handlers.

Comparison in [01_tool_decorator_and_structured_tool.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/01_tool_decorator_and_structured_tool.py):
```python
from langchain_core.tools import BaseTool, StructuredTool, tool

# 1. Decorator
@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)

# 2. StructuredTool
multiply_tool = StructuredTool.from_function(func=multiply, name="multiply", description="...")

# 3. BaseTool subclassing
class ReverseStringTool(BaseTool):
    name: str = "reverse_string"
    description: str = "..."
    def _run(self, text: str) -> str: return text[::-1]
```

---

## 2. Input Schema Validation with Pydantic
For tools with multiple parameters, it is best practice to pass an explicit Pydantic model to **`args_schema`**. This generates detailed parameter-level JSON schemas that steer models to write inputs correctly.

In [02_tool_schemas_pydantic.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/02_tool_schemas_pydantic.py):
```python
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    city: str = Field(description="the city to look up weather for, e.g. 'Paris'")
    units: str = Field(default="celsius", description="temperature units")

@tool(args_schema=WeatherQuery)
def get_weather(city: str, units: str = "celsius") -> str:
    """Look up the current weather for a city."""
    ...
```

---

## 3. Built-In Toolkits
LangChain includes pre-packaged sets of tools called **toolkits** that expose common external tools:
- **`SQLDatabaseToolkit`**: Exposes queries, schemas, and verification check functions on SQL databases.
- **`PythonREPLTool`**: Executes raw Python statements inside a persistent environment (Caution: execute only in sandboxed runtimes).
- **`RequestsGetTool`**: Issues outbound HTTP requests (requires setting `allow_dangerous_requests=True`).
- **`TavilySearchResults`**: Queries the Tavily web search endpoint to return short document summaries.

Examples in [03_builtin_toolkits.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/03_builtin_toolkits.py).

---

## 4. Tool Calling Protocol & Formatting
The tool execution workflow follows a sequence (detailed in [04_tool_calling_format.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/04_tool_calling_format.py)):
1. **Schema binding**: Bind tools to a model using `.bind_tools([get_stock_price])`. This passes the schema specifications in the API request.
2. **Model decision**: The model decides if a tool is needed. If so, it returns an empty-content `AIMessage` containing a `tool_calls` list with unique IDs.
3. **Execution**: Your application intercepts the `tool_calls`, invokes the requested tool locally, and returns the output wrapped in a `ToolMessage` with a matching `tool_call_id`.
4. **Final response**: Feed the conversation history containing the `ToolMessage` back to the model so it can formulate a final answer.

```python
# Bind tools
model_with_tools = model.bind_tools([get_stock], tool_choice="get_stock") # tool_choice forces usage
```

---

## 5. Error Handling & ToolNode Execution Loop
When a model issues multiple tool requests, a tool may fail or raise exceptions. Instead of crashing, your program should capture the exception and pass the error back as the tool output. This allows the model to observe the issue and correct its arguments in a subsequent turn.

Comparison in [05_toolnode_execution.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/05_toolnode_execution.py):
- **Hand-rolled execution loop**:
```python
def execute_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    results = []
    for call in ai_message.tool_calls:
        try:
            output = tools[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        except Exception as exc:
            results.append(ToolMessage(content=f"Error: {exc}", tool_call_id=call["id"]))
    return results
```
- **LangGraph Prebuilt Node**:
LangGraph encapsulates this entire process inside the prebuilt **`ToolNode`** component:
```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([divide, lookup_capital])
```

---

## 6. Advanced: Injected Tools Arguments & Split Artifacts

### InjectedToolArg
Some tool parameters are sensitive or context-dependent (e.g. database connections or user identity IDs). To prevent the model from hallucinating or forging these values, annotate them with **`InjectedToolArg`**. This hides the fields from the schema sent to the model while allowing you to pass them manually at call-time.

```python
from typing import Annotated
from langchain_core.tools import InjectedToolArg, tool

@tool
def get_orders(user_id: Annotated[str, InjectedToolArg], status: str) -> str:
    ...
```

### returning Content and Artifacts
Setting `response_format="content_and_artifact"` on a tool allows it to return a tuple `(content, artifact)` (demonstrated in [06_injected_args_and_artifacts.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/06_injected_args_and_artifacts.py#L34-L39)):
- **`content`**: A short text summary passed to the model.
- **`artifact`**: A rich object (e.g., dictionary, image, dataframe) retained by your application for rendering in a UI or performing database updates.
```python
@tool(response_format="content_and_artifact")
def run_sql(query: str) -> tuple[str, dict]:
    return "Query executed.", {"rows": [...]}
```

---

## 7. Sandboxing Tool Execution & Least-Privilege Tools
When letting models run code, call databases, or execute command-line shell utilities, strict constraints are required to protect the host environment.
- **Path Traversal Defenses**: Standard checks preventing the model from specifying parameters containing path escapes (`..`, `/`).
- **Subprocess Isolation**: Spawning separate processes with low privileges.
- **Timeout Controls**: Enforcing execution limits to avoid infinite loop resource exhaustion.

See [07_tool_sandboxing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/07_tool_sandboxing.py) for examples.
