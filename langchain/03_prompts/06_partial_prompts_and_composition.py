"""
Partial Prompts and Prompt Composition/Piping
================================================
- Partial prompts: pre-fill some template variables ahead of time (e.g. a
  fixed "today's date" or "system persona"), leaving the rest to be filled
  in later when the chain actually runs.
- Prompt composition: combine multiple smaller prompt templates into one,
  either by string concatenation (`+`) or by piping.
"""

from datetime import date
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# --- Partial with a static value ---
template = PromptTemplate.from_template("Today is {today}. {question}")
partial_template = template.partial(today=str(date.today()))

print(partial_template.format(question="What should I plan?"))
# -> "Today is 2026-08-01. What should I plan?"

# --- Partial with a function (evaluated lazily, at format-time) ---
def _get_today() -> str:
    return str(date.today())

dynamic_partial = template.partial(today=_get_today)
print(dynamic_partial.format(question="Any meetings?"))

# --- Composing PromptTemplates with string `+` ---
intro = PromptTemplate.from_template("You are a helpful assistant named {name}.")
task = PromptTemplate.from_template("Your task: {task}")
combined = intro + "\n" + task
print(combined.format(name="Ada", task="explain recursion"))

# --- Composing ChatPromptTemplates ---
system = ChatPromptTemplate.from_messages([("system", "You are {persona}.")])
human = ChatPromptTemplate.from_messages([("human", "{question}")])
chat_combined = system + human
print(chat_combined.format_messages(persona="a Python tutor", question="What is a decorator?"))
