"""
PROMPTTEMPLATE AND CHATPROMPTTEMPLATE
========================================
Prompt templates let you parameterize prompts with `{variable}` placeholders
instead of hand-building strings. There are two main flavors:

    PromptTemplate     -> produces a single plain-text string (for legacy LLMs
                           or any place you need raw text)
    ChatPromptTemplate  -> produces a list of messages (for ChatModels),
                           built from role/template pairs

Both are Runnables, so they compose directly with models via the `|` operator.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama

# --- PromptTemplate: plain string output ----------------------------------
text_prompt = PromptTemplate.from_template(
    "Write a {length}-sentence summary of the topic: {topic}."
)

rendered = text_prompt.invoke({"length": "two", "topic": "photosynthesis"})
print(rendered.to_string())
# -> "Write a two-sentence summary of the topic: photosynthesis."

# --- ChatPromptTemplate: list-of-messages output ---------------------------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain {concept} to a beginner."),
    ]
)

rendered_messages = chat_prompt.invoke(
    {"domain": "astronomy", "concept": "black holes"}
)
print(rendered_messages.to_messages())
# -> [SystemMessage(content='You are a helpful astronomy expert.'),
#     HumanMessage(content='Explain black holes to a beginner.')]

# --- Composing a template directly into a chain with | --------------------
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
chain = chat_prompt | model

response = chain.invoke({"domain": "cooking", "concept": "deglazing"})
print(response.content)
# -> "Deglazing means adding liquid to a hot pan to lift up the browned bits..."

# --- Templates validate their expected input variables --------------------
print(chat_prompt.input_variables)
# -> ['domain', 'concept']

# Missing a variable at invoke time raises a KeyError, which is useful for
# catching prompt/data mismatches early instead of silently sending a
# malformed prompt to the model.
