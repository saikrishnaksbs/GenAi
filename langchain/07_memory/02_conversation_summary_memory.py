"""
CONVERSATION SUMMARY MEMORY
=============================
ConversationSummaryMemory keeps the chat history as a running natural-language
summary instead of verbatim turns. Every time you add a new exchange, it
calls the LLM again to fold the new turn into the existing summary. This
keeps token usage roughly constant regardless of conversation length, at the
cost of a summarization LLM call on every turn (and potential loss of detail).

ConversationSummaryBufferMemory is a hybrid: it keeps recent messages verbatim
in a buffer, and only summarizes older messages once the buffer exceeds a
token limit. This gives you exact recent context plus a compressed history.
"""

from langchain.memory import ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- ConversationSummaryMemory ---
# Requires an LLM because summarizing IS an LLM call.
summary_memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=summary_memory)

conversation.predict(input="I'm planning a trip to Japan in October.")
conversation.predict(input="I want to visit Kyoto and Osaka, budget is $3000.")

print(summary_memory.buffer)
# -> "The human is planning a trip to Japan in October, wants to visit Kyoto
#     and Osaka, with a budget of $3000."  (a compressed paraphrase, not verbatim)

print(summary_memory.load_memory_variables({}))
# -> {"history": "<the running summary string>"}


# --- ConversationSummaryBufferMemory ---
# max_token_limit controls when older turns get rolled into the summary.
summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,  # once buffered messages exceed ~100 tokens, oldest get summarized
)

buffered_conversation = ConversationChain(llm=llm, memory=summary_buffer_memory)

buffered_conversation.predict(input="Let's talk about the history of the Roman Empire.")
buffered_conversation.predict(input="Who was Augustus?")
buffered_conversation.predict(input="What about Julius Caesar?")
# Early turns get summarized once the token limit is exceeded; recent turns
# stay verbatim in `moving_summary_buffer` + `chat_memory`.

print(summary_buffer_memory.load_memory_variables({}))
# -> {"history": "System: <summary of earlier turns>\nHuman: What about Julius
#     Caesar?\nAI: ..."}  (mix of summary + verbatim recent messages)

# Inspect just the running summary text
print(summary_buffer_memory.moving_summary_buffer)
# -> "The human and AI discussed the history of the Roman Empire, including
#     Augustus's rise to power..."
