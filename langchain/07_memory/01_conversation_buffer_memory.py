"""
CONVERSATION BUFFER MEMORY
===========================
ConversationBufferMemory stores the entire chat history verbatim and injects
it into the prompt on every call. It is the simplest memory type: nothing is
summarized or dropped, so the prompt grows unbounded as the conversation
continues.

ConversationBufferWindowMemory is the same idea but only keeps the last `k`
exchanges, which caps token growth at the cost of losing older context.

These are legacy APIs (langchain.memory) — in modern LangChain, memory is
handled with RunnableWithMessageHistory instead. They're shown here because
a lot of production code still uses them.
"""

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- ConversationBufferMemory: keeps everything ---
buffer_memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=buffer_memory, verbose=False)

conversation.predict(input="Hi, I'm Sai. I live in Bangalore.")
# -> "Hello Sai! Nice to meet you..."

conversation.predict(input="What city did I say I live in?")
# -> "You said you live in Bangalore."

print(buffer_memory.buffer)
# -> "Human: Hi, I'm Sai. I live in Bangalore.\nAI: Hello Sai!...\nHuman: What city..."

# load_memory_variables returns the dict the chain injects into the prompt
print(buffer_memory.load_memory_variables({}))
# -> {"history": "Human: Hi, I'm Sai...\nAI: ..."}


# --- ConversationBufferWindowMemory: keeps only the last k turns ---
window_memory = ConversationBufferWindowMemory(k=2)  # remember last 2 exchanges only

windowed_conversation = ConversationChain(llm=llm, memory=window_memory)

windowed_conversation.predict(input="My favorite color is blue.")
windowed_conversation.predict(input="My favorite food is dosa.")
windowed_conversation.predict(input="My favorite sport is cricket.")
# By now, the first exchange (favorite color) has been pushed out of the window

print(window_memory.load_memory_variables({}))
# -> only the last 2 Human/AI exchanges appear (food + sport), color is gone

# You can also manually save context without an LLMChain wrapper
manual_memory = ConversationBufferMemory()
manual_memory.save_context({"input": "Hi there"}, {"output": "Hello! How can I help?"})
print(manual_memory.load_memory_variables({}))
# -> {"history": "Human: Hi there\nAI: Hello! How can I help?"}
