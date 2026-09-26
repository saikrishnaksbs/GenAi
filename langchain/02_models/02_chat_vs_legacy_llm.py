"""
CHAT MODEL VS LEGACY LLM INTERFACE
====================================
LangChain historically supported two kinds of language model wrappers:

    - `LLM`       : takes a plain string prompt, returns a plain string.
                    Modeled on old-style completion APIs (e.g. text-davinci-003).
    - `ChatModel` : takes a list of messages, returns a single AIMessage.
                    Modeled on modern chat-based APIs (OpenAI chat, Claude, etc.)

Almost all modern providers only expose chat endpoints now, so `ChatModel`
subclasses (ChatOpenAI, ChatAnthropic, ...) are the recommended interface.
The legacy `LLM` interface still exists for older/completion-style models
and is useful to recognize in older tutorials and codebases.
"""

from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama

# --- Legacy LLM interface: string in, string out -----------------------
legacy_llm = Ollama(model="qwen2.5:1.5b", temperature=0)

legacy_result = legacy_llm.invoke("Write a haiku about databases.")
print(legacy_result)
# -> "Rows and columns hum\nqueries whisper through indexes\nsilence, then an answer"
print(type(legacy_result))
print()
# -> <class 'str'>

# --- Modern ChatModel interface: messages in, AIMessage out ------------
chat_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

chat_result = chat_model.invoke("Write a haiku about databases.")
print(chat_result)
print()

# -> AIMessage(content="Rows and columns hum\n...", response_metadata={...})
print(type(chat_result))
print()

# -> <class 'langchain_core.messages.ai.AIMessage'>

# ChatModel.invoke() also accepts a plain string as shorthand -- LangChain
# wraps it into a single HumanMessage under the hood.
print(chat_result.content)
print()

# -> "Rows and columns hum\nqueries whisper through indexes\nsilence, then an answer"

# Both LLM and ChatModel implement the Runnable interface, so .stream(),
# .batch(), and LCEL piping (`prompt | model`) work identically on either.
# The key practical difference is the shape of input/output:
#   LLM:       str            -> str
#   ChatModel: list[Message]  -> AIMessage
