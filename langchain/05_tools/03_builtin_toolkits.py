"""
BUILT-IN TOOLKITS
====================
LangChain ships several pre-built toolkits so you don't have to hand-write
tools for common integrations. A "toolkit" is just a class that bundles a
related set of Tool objects (e.g. all the SQL operations for a database).

This file shows four commonly used ones:
  - SQLDatabaseToolkit: query/inspect a SQL database
  - PythonREPLTool: execute Python code
  - requests toolkit: make HTTP calls
  - A search API tool (Tavily) as an example of a hosted search integration
"""

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_community.chat_models import ChatOllama

# --- SQL toolkit -----------------------------------------------------------
db = SQLDatabase.from_uri("sqlite:///example.db")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

print([t.name for t in sql_tools])
# -> ["sql_db_query", "sql_db_schema", "sql_db_list_tables", "sql_db_query_checker"]

list_tables_tool = next(t for t in sql_tools if t.name == "sql_db_list_tables")
print(list_tables_tool.invoke(""))
# -> "customers, orders, products"


# --- Python REPL tool --------------------------------------------------------
python_tool = PythonREPLTool()
# Executes arbitrary Python in a persistent namespace — powerful but only
# safe in sandboxed/trusted environments, since it runs real code.
print(python_tool.invoke("print(sum(range(1, 11)))"))
# -> "55\n"


# --- requests toolkit ----------------------------------------------------
requests_wrapper = TextRequestsWrapper(headers={"User-Agent": "langchain-example"})
get_tool = RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True)
# `allow_dangerous_requests=True` is required because this tool lets the
# model trigger arbitrary outbound HTTP GET requests.
try:
    # Attempt to request httpbin.org which is a reliable public echo/testing service.
    # Note: If you don't have internet access, this call will fail gracefully.
    response_text = get_tool.invoke("https://httpbin.org/get")
    print("HTTP GET Response:")
    print(response_text[:100] + "...")
except Exception as e:
    print(f"HTTP GET request failed: {e}")
    print("Fallback mock response for demonstration:")
    response_text = '{"status": "ok", "uptime_seconds": 123456}'
    print(response_text[:50])


# --- Hosted search API tool (Tavily) --------------------------------------
# Requires TAVILY_API_KEY in the environment. This is a common "search the
# web" tool used in agent examples since it returns clean, LLM-ready results.
search_tool = None
try:
    search_tool = TavilySearchResults(max_results=3)
    results = search_tool.invoke("latest LangChain release notes")
    print("Tavily Search Results:")
    print(results)
except Exception as e:
    print(f"Tavily search tool skipped/failed (requires TAVILY_API_KEY env var): {e}")

# Toolkits and individual tools compose the same way — you pass a flat
# list of tools to an agent regardless of which toolkit they came from.
additional_tools = [python_tool, get_tool]
if search_tool is not None:
    additional_tools.append(search_tool)
all_tools = sql_tools + additional_tools
print(f"Total tools loaded: {len(all_tools)}")

