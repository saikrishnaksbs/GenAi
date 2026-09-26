# LangChain Tools Cheat Sheet & Topic Shortcuts

A quick-reference guide to help you recall and implement all tool patterns in the [05_tools](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools) module.

---

## 1. Defining Tools
*File:* [01_tool_decorator_and_structured_tool.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/01_tool_decorator_and_structured_tool.py)

Three primary patterns to create a tool:

```python
from langchain_core.tools import BaseTool, StructuredTool, tool

# Pattern A: @tool Decorator (Simple, clean)
@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""  # Used as tool description
    return len(word)

# Pattern B: StructuredTool (Wrap external/library functions)
multiply_tool = StructuredTool.from_function(
    func=multiply, name="multiply", description="Multiply two integers."
)

# Pattern C: Subclassing BaseTool (Stateful, custom init/async overrides)
class ReverseStringTool(BaseTool):
    name: str = "reverse_string"
    description: str = "Reverse characters in the text."
    
    def _run(self, text: str) -> str:
        return text[::-1]
    async def _arun(self, text: str) -> str:
        return self._run(text)
```

---

## 2. Input Schema Validation with Pydantic
*File:* [02_tool_schemas_pydantic.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/02_tool_schemas_pydantic.py)

Forces the LLM to output valid arguments using Pydantic validation schemas.

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WeatherQuery(BaseModel):
    city: str = Field(description="The city name, e.g. 'Paris'")
    units: str = Field(default="celsius", description="celsius or fahrenheit")

@tool(args_schema=WeatherQuery)
def get_weather(city: str, units: str = "celsius") -> str:
    """Look up current weather."""
    return f"Weather in {city}: 18 {units}"
```

---

## 3. Built-In Toolkits
*File:* [03_builtin_toolkits.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/03_builtin_toolkits.py)

Common ready-made tools prepackaged in LangChain:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.tools import TavilySearchResults

# SQL Database Tooling
sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

# Python Code Sandbox (Unsafe - run inside Docker/Sandboxes only!)
python_tool = PythonREPLTool()

# Outbound HTTP GET calls
get_tool = RequestsGetTool(requests_wrapper=wrapper, allow_dangerous_requests=True)

# Web Search
search_tool = TavilySearchResults(max_results=3)
```

---

## 4. Tool Calling Protocol
*File:* [04_tool_calling_format.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/04_tool_calling_format.py)

Understanding the standard binding, request, and feedback loop:

```python
# 1. Bind tools to the model
model_with_tools = model.bind_tools([get_stock_price], tool_choice="get_stock_price")

# 2. Invoke model to get AIMessage with tool_calls
ai_msg = model_with_tools.invoke([HumanMessage("Stock price of AAPL?")])
# ai_msg.tool_calls -> [{'name': 'get_stock_price', 'args': {'ticker': 'AAPL'}, 'id': 'call_123', ...}]

# 3. Invoke tool and return a ToolMessage
output = get_stock_price.invoke(ai_msg.tool_calls[0]["args"])
tool_msg = ToolMessage(content=str(output), tool_call_id=ai_msg.tool_calls[0]["id"])

# 4. Feed back to model for the final response
final = model_with_tools.invoke([HumanMessage(...), ai_msg, tool_msg])
```

---

## 5. Tool Execution & Errors
*File:* [05_toolnode_execution.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/05_toolnode_execution.py)

Convert runtime tool errors to `ToolMessage` contents instead of letting the program crash.

```python
# Hand-rolled loop
def execute_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    results = []
    for call in ai_message.tool_calls:
        try:
            output = tools[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        except Exception as e:
            results.append(ToolMessage(content=f"Error: {e}", tool_call_id=call["id"]))
    return results

# LangGraph Shortcut
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([divide, lookup_capital])
node_result = tool_node.invoke({"messages": [ai_message]})
```

---

## 6. Injected Arguments & Artifacts
*File:* [06_injected_args_and_artifacts.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/06_injected_args_and_artifacts.py)

Advanced handling of sensitive variables and heavy response data.

```python
from typing import Annotated
from langchain_core.tools import InjectedToolArg, tool

# InjectedToolArg: Hides argument from LLM schema; you pass it at invoke time
@tool
def get_orders(user_id: Annotated[str, InjectedToolArg], status: str) -> str:
    return f"Orders for {user_id}"

# Tool call-time injection
get_orders.invoke({"status": "shipped", "user_id": "u_42"})

# content_and_artifact: Separates simple content (for LLM) from raw/heavy artifacts (for UI/App)
@tool(response_format="content_and_artifact")
def run_sql(query: str) -> tuple[str, dict]:
    return "Query successful", {"rows": [1, 2, 3]}

# Result has message.content and message.artifact
tool_msg = run_sql.invoke({"name": "run_sql", "args": {"query": "..."}, "id": "call_1"})
# tool_msg.content  -> "Query successful"
# tool_msg.artifact -> {"rows": [1, 2, 3]}
```

---

## 7. Sandboxing Tool Execution
*File:* [07_tool_sandboxing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/07_tool_sandboxing.py)

Shields your host environment from malicious or runaway inputs.

```python
import subprocess
import re

# 1. Path Traversal & Shell Input Validation
def safe_filename(filename: str):
    if "/" in filename or ".." in filename:
        raise ValueError("Invalid path")
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
        raise ValueError("Forbidden characters")

# 2. Subprocess Isolation with Timeouts (Anti-Infinite Loop)
try:
    res = subprocess.run(
        ["python3", "-c", user_code],
        capture_output=True,
        timeout=2.0  # Limit duration
    )
except subprocess.TimeoutExpired:
    print("Execution exceeded timeout limit.")
```

---

## 8. ToolNode Error Handling
*File:* [08_toolnode_error_handling.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/05_tools/08_toolnode_error_handling.py)

How LangGraph's prebuilt `ToolNode` manages tool-level errors under the hood.

* **Behavior**: If any tool throws an exception, `ToolNode` catches it, creates a `ToolMessage` with `content="Error: <Exception info>"`, and appends it to the messages state.
* **LLM Correction turn**: By receiving the error back in its context, the LLM is prompted to correct its input parameters in the subsequent turn instead of halting execution.
