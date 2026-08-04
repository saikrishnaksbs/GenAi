"""
STR OUTPUT PARSER
=================
`StrOutputParser` is the simplest output parser in LangChain. It takes the
raw `AIMessage` returned by a chat model and extracts just the `.content`
string, discarding metadata like token usage or tool calls.

It's the default "glue" at the end of most LCEL chains: prompt | model | parser.
Without it you'd have to manually call `.content` on every model response.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("Give a one-sentence fact about {topic}.")

# LCEL chain: each Runnable's output feeds the next one's input.
chain = prompt | model | parser

result = chain.invoke({"topic": "octopuses"})
print(result)
# -> "Octopuses have three hearts and blue blood."

# Without the parser, `chain.invoke` would return an AIMessage object,
# and you'd need `response.content` to get the plain string:
raw_chain = prompt | model
raw_response = raw_chain.invoke({"topic": "octopuses"})
print(type(raw_response))
# -> <class 'langchain_core.messages.ai.AIMessage'>
print(raw_response.content == result)
# -> True (StrOutputParser just unwraps .content)

# StrOutputParser also works standalone on an already-generated message.
from langchain_core.messages import AIMessage

manual_message = AIMessage(content="Paris is the capital of France.")
print(parser.invoke(manual_message))
# -> "Paris is the capital of France."

# It streams token-by-token just like the model does, since it implements
# the Runnable streaming protocol transparently.
for chunk in chain.stream({"topic": "the moon"}):
    print(chunk, end="", flush=True)
# -> streams words as they arrive, e.g. "The" "Moon" " is" " drifting" ...
