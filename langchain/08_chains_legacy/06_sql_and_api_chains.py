"""
SQLDatabaseChain and APIChain
================================
Legacy chains that let an LLM translate natural language into a SQL query
(and execute it) or into an API call (and make it), then answer using the
result. Modern LangChain generally prefers doing this with tool-calling
agents (see 09_agents_legacy), but these chains are still found in older
codebases and are simpler for constrained, single-step use cases.
"""

from langchain_community.utilities import SQLDatabase
from langchain_community.chains import SQLDatabaseChain  # legacy import path
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- SQLDatabaseChain: NL question -> SQL query -> execute -> NL answer ---
db = SQLDatabase.from_uri("sqlite:///example.db")
sql_chain = SQLDatabaseChain.from_llm(model, db, verbose=True)

answer = sql_chain.invoke({"query": "How many users signed up last month?"})
print(answer["result"])
# Under the hood: model writes a SELECT statement, db.run() executes it,
# model then converts the raw rows into a natural-language answer.

# --- APIChain: NL question -> API call (using a provided OpenAPI spec) -> NL answer ---
from langchain.chains import APIChain

api_docs = """
BASE URL: https://api.open-meteo.com
Endpoint: GET /v1/forecast?latitude={lat}&longitude={lon}&current_weather=true
Returns current weather for the given coordinates.
"""

api_chain = APIChain.from_llm_and_api_docs(
    model,
    api_docs,
    verbose=True,
    limit_to_domains=["https://api.open-meteo.com"],  # safety: restrict callable domains
)

weather = api_chain.invoke({"question": "What's the weather at latitude 12.9, longitude 77.6?"})
print(weather["output"])
