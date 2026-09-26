"""
SQL AND API CHAINS (Modern LCEL Replacement for SQLDatabaseChain & APIChain)
=============================================================================
Legacy LangChain used `SQLDatabaseChain` and `APIChain` to query databases and call APIs.
In modern LangChain, these monolithic chains are deprecated in favor of:
1. `create_sql_query_chain`: Formulates a SQL query from natural language.
2. `QuerySQLDatabaseTool`: Executes the generated SQL query.
3. Explicit LCEL composition: Maps prompts and HTTP requests directly, giving you complete
   visibility and custom logic options (e.g. safety validation).
"""

import os
import sqlite3
import requests
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain

# --- Initialize SQLite Database ---
db_path = "example.db"
if not os.path.exists(db_path):
    print("Initializing example SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            signup_date TEXT
        )
    """)
    cursor.executemany("""
        INSERT INTO users (username, signup_date) VALUES (?, ?)
    """, [
        ("alice", "2026-07-15"),
        ("bob", "2026-07-20"),
        ("charlie", "2026-08-01"),
    ])
    conn.commit()
    conn.close()

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- 1. Modern SQL RAG Chain ---
# Goal: Convert a natural language question to a SQL query, execute it against the SQLite
# database, and format the raw result back into a natural language response.
print("\n=== Running SQL RAG Chain ===")

# 1. create_sql_query_chain:
#    Analyzes the database schema (table definitions, columns, types) and instructs the
#    LLM to output a syntactically correct SQL query (e.g., SELECT ... FROM ...).
write_query = create_sql_query_chain(llm, db)

# 2. QuerySQLDatabaseTool:
#    A helper tool that connects to the SQL database, runs the generated SQL query string,
#    and returns the result (rows) as a plain text string.
execute_query = QuerySQLDatabaseTool(db=db)

# 3. answer_prompt:
#    Instructs the LLM to write a final human-friendly response by combining:
#    - The original question.
#    - The SQL query that was run.
#    - The SQL result (data rows).
answer_prompt = PromptTemplate.from_template(
    "Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n\n"
    "Question: {question}\n"
    "SQL Query: {query}\n"
    "SQL Result: {result}\n\n"
    "Answer: "
)

# 4. LCEL Pipeline Construction:
#    - RunnablePassthrough.assign(query=write_query): Adds a new key "query" to the input dict by calling write_query.
#    - RunnablePassthrough.assign(result=...): Adds a "result" key by executing the SQL query generated in the previous step.
#    - The fully populated dictionary {"question": ..., "query": ..., "result": ...} is passed to answer_prompt.
#    - The prompt is sent to the LLM, and the output is parsed as a string.
sql_rag_chain = (
    RunnablePassthrough.assign(query=write_query)
    | RunnablePassthrough.assign(result=lambda x: execute_query.invoke({"query": x["query"]}))
    | answer_prompt
    | llm
    | StrOutputParser()
)

sql_response = sql_rag_chain.invoke({"question": "How many users signed up last month (July 2026)?"})
print("Result:", sql_response)


# --- 2. Modern API Chain ---
# Goal: Translate a user's question into an HTTP GET URL request based on API documentation,
# execute the request, and generate a natural language summary of the returned API payload.
print("\n=== Running API Chain ===")

# Define documentation for the API so the LLM knows how to structure the parameters
api_docs = """
BASE URL: https://api.open-meteo.com
Endpoint: GET /v1/forecast?latitude={lat}&longitude={lon}&current_weather=true
Returns current weather for the given coordinates.
"""

# Prompt to translate a natural language question into a structured GET API endpoint URL
url_prompt = ChatPromptTemplate.from_template(
    "Based on the following API docs, generate the exact URL to call for the user's question.\n"
    "Respond with ONLY the URL and nothing else.\n\n"
    "API Docs:\n{api_docs}\n\n"
    "Question: {question}"
)

# A simple chain that outputs the exact URL string
url_generator = url_prompt | llm | StrOutputParser()

# HTTP API caller function with domain validation (security guardrail)
def call_api(url: str) -> str:
    """Executes the HTTP GET request after validating the domain to prevent SSRF or unauthorized calls."""
    url = url.strip()
    allowed_domain = "https://api.open-meteo.com"
    
    # Safety Check: Limit API execution to authorized external domains
    if not url.startswith(allowed_domain):
        return f"Error: Domain not allowed. Must start with {allowed_domain}"
    
    try:
        response = requests.get(url, timeout=10)
        return response.text
    except Exception as e:
        return f"Error executing API call: {str(e)}"

# Final prompt to compile the original question, URL called, and the raw API response into a final answer
api_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user question based on the API response details."),
    ("human", "Question: {question}\nAPI URL called: {url}\nAPI Response: {response}"),
])

# LCEL composition:
# 1. Takes input {"question": ..., "api_docs": ...}
# 2. Generates and assigns the URL.
# 3. Executes the API call using a RunnableLambda and assigns the output to "response".
# 4. Feeds the accumulated dictionary into the final answering prompt, LLM, and parser.
api_chain = (
    RunnablePassthrough.assign(url=url_generator)
    | RunnablePassthrough.assign(response=RunnableLambda(lambda x: call_api(x["url"])))
    | api_answer_prompt
    | llm
    | StrOutputParser()
)

api_response = api_chain.invoke({"question": "What's the weather at latitude 12.9, longitude 77.6?", "api_docs": api_docs})
print("Result:", api_response)
