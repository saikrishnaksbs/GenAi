"""
CONVERSATION TOKEN BUFFER MEMORY
===================================
ConversationTokenBufferMemory is like ConversationBufferWindowMemory, but
instead of keeping the last `k` exchanges, it keeps as many recent messages
as fit within a token budget (`max_token_limit`). Whenever adding a new
message would push the buffer over that limit, the oldest messages are
dropped one at a time until it fits again.

This is useful when messages vary a lot in length — a fixed message count
(window memory) can either waste context or blow past the model's limit,
whereas a token-based cutoff adapts to actual message size.

Note: it needs an `llm` reference purely to count tokens with that model's
tokenizer, not to generate text.
"""

from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

token_memory = ConversationTokenBufferMemory(
    llm=llm,          # used only for token counting via llm.get_num_tokens
    max_token_limit=60,  # keep roughly the last 60 tokens' worth of messages
)

conversation = ConversationChain(llm=llm, memory=token_memory)

conversation.predict(input="My name is Sai and I work as a software engineer.")
conversation.predict(input="I mainly work with Python and distributed systems.")
conversation.predict(input="Lately I've been exploring LangChain and RAG pipelines.")
# As the buffer exceeds ~60 tokens, the earliest messages (e.g. the name intro)
# get evicted first.

print(token_memory.load_memory_variables({}))
# -> {"history": "Human: I mainly work with Python...\nAI: ...\nHuman: Lately
#     I've been exploring LangChain..."}  (oldest turn may already be gone)

# You can inspect the raw message list directly
for message in token_memory.chat_memory.messages:
    print(type(message).__name__, "->", message.content)
# -> HumanMessage -> I mainly work with Python and distributed systems.
# -> AIMessage -> ...
# -> HumanMessage -> Lately I've been exploring LangChain and RAG pipelines.
# -> AIMessage -> ...

# Manually saving context still respects the token limit on the next trim
token_memory.save_context(
    {"input": "One more fact: I prefer dark mode everywhere."},
    {"output": "Noted! Dark mode it is."},
)
print(len(token_memory.chat_memory.messages))
# -> trimmed to fit max_token_limit, so old messages may have been dropped
