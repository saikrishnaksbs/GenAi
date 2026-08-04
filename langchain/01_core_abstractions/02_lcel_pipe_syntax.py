"""
LCEL PIPE SYNTAX (RunnableSequence)
=====================================
LCEL (LangChain Expression Language) lets you compose Runnables with the
`|` operator, exactly like a Unix pipe. `a | b` builds a `RunnableSequence`
where the output of `a` becomes the input of `b`.

This is the idiomatic way to build chains in modern LangChain, replacing
the older `LLMChain` class.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence for a {audience}."
)
model = ChatOllama(model="qwen2.5:1.5b")
parser = StrOutputParser()

# The pipe operator chains Runnables into a RunnableSequence.
chain = prompt | model | parser

result = chain.invoke({"topic": "LCEL", "audience": "beginner"})
print(result)
# -> "LCEL is a way to chain LangChain components together using the | operator."

# Under the hood this is equivalent to:
from langchain_core.runnables import RunnableSequence

explicit_chain = RunnableSequence(first=prompt, middle=[model], last=parser)

# Chains can be arbitrarily long — each step's output type must match the next
# step's expected input type.
longer_chain = prompt | model | parser | (lambda s: s.strip().upper())
print(longer_chain.invoke({"topic": "chains", "audience": "engineer"}))
