"""
RunnableParallel and RunnableBranch
=====================================
RunnableParallel runs multiple Runnables concurrently on the same input and
returns a dict of their outputs. RunnableBranch routes input to different
Runnables based on conditions (like an if/elif/else for chains).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")
parser = StrOutputParser()

# --- RunnableParallel: fan-out, run both branches on the same input ---
summarize_prompt = ChatPromptTemplate.from_template("Summarize: {text}")
sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment of: {text}")

parallel_chain = RunnableParallel(
    summary=summarize_prompt | model | parser,
    sentiment=sentiment_prompt | model | parser,
)

result = parallel_chain.invoke({"text": "LangChain makes building LLM apps easier."})
print(result)
# -> {"summary": "...", "sentiment": "..."}

# --- RunnableBranch: route based on a condition function ---
def is_code_question(input_dict) -> bool:
    return "code" in input_dict["question"].lower()

code_chain = ChatPromptTemplate.from_template(
    "Answer this coding question precisely: {question}"
) | model | parser

general_chain = ChatPromptTemplate.from_template(
    "Answer this general question: {question}"
) | model | parser

branch = RunnableBranch(
    (is_code_question, code_chain),   # (condition, runnable) pairs, checked in order
    general_chain,                    # default/fallback runnable
)

print(branch.invoke({"question": "How do I write code for a for-loop in Python?"}))
print(branch.invoke({"question": "What is the capital of France?"}))
