"""
MESSAGESPLACEHOLDER
=====================
`MessagesPlaceholder` reserves a spot inside a ChatPromptTemplate for a
*list* of messages supplied at invoke time, rather than a single templated
string. This is exactly what you need for injecting conversation history
(or few-shot examples) into an otherwise fixed prompt skeleton.

Without it, you'd have to manually splice message lists together yourself
every time you build a prompt.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),  # slot for past turns
        ("human", "{input}"),
    ]
)

# The chat_history variable is filled with an actual list of Message objects,
# built up as the conversation progresses (e.g. from memory or a message store).
chat_history = [
    HumanMessage(content="I ordered a lamp but it hasn't arrived."),
    AIMessage(content="I'm sorry to hear that! Can you share your order number?"),
    HumanMessage(content="It's ORD-48213."),
    AIMessage(content="Thanks, I found it -- it's out for delivery today."),
]

rendered = prompt.invoke({"chat_history": chat_history, "input": "Great, thank you!"})
print(rendered.to_messages())
# -> [SystemMessage(content='You are a friendly customer support agent.'),
#     HumanMessage(content="I ordered a lamp but it hasn't arrived."),
#     AIMessage(content="I'm sorry to hear that! Can you share your order number?"),
#     HumanMessage(content='It's ORD-48213.'),
#     AIMessage(content="Thanks, I found it -- it's out for delivery today."),
#     HumanMessage(content='Great, thank you!')]

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
chain = prompt | model

response = chain.invoke({"chat_history": chat_history, "input": "Great, thank you!"})
print(response.content)
# -> "You're welcome! Let us know if there's anything else we can help with."

# --- MessagesPlaceholder with an empty history is perfectly valid --------
response = chain.invoke({"chat_history": [], "input": "Hi, is anyone there?"})
print(response.content)
# -> "Hello! Yes, I'm here and happy to help."

# --- Optional placeholders: allow omitting the key entirely ---------------
optional_placeholder = MessagesPlaceholder(variable_name="chat_history", optional=True)
# When optional=True, invoking without a "chat_history" key simply skips it
# instead of raising a missing-variable error.
