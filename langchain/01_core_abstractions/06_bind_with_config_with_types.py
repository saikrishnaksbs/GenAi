"""
bind(), with_config(), with_types()
======================================
- .bind(**kwargs): permanently attaches extra keyword args to every call of a
  Runnable (e.g. always pass `stop=["\\n"]` to a model, or fix a tool list).
- .with_config(**kwargs): permanently attaches RunnableConfig fields (tags,
  metadata, callbacks) to a Runnable — see 05_runnable_config.py.
- .with_types(): overrides the input/output type LangChain infers for a
  Runnable, mainly useful for schema validation and LangServe playgrounds.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")

# .bind() locks in a keyword arg for every future call — here, a stop sequence.
model_with_stop = model.bind(stop=["\n\n"])

chain = ChatPromptTemplate.from_template("List 3 facts about {topic}") | model_with_stop | StrOutputParser()
print(chain.invoke({"topic": "the moon"}))

# .bind() is also how tools get attached before with_structured_output/bind_tools
# existed as dedicated helpers:
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

model_with_tool = model.bind(tools=[{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}])

# .with_config() — see 05_runnable_config.py for full detail.
tagged_chain = chain.with_config(tags=["facts-chain"], run_name="list_facts")

# .with_types() — override the declared input/output schema (advanced, mostly
# used for exposing chains via LangServe with a strict client-facing schema).
from pydantic import BaseModel

class TopicInput(BaseModel):
    topic: str

typed_chain = chain.with_types(input_type=TopicInput)
