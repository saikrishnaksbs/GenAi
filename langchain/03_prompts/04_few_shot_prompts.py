"""
FEW-SHOT PROMPT TEMPLATES
============================
Few-shot prompting shows the model several example input/output pairs
before asking it to handle a new input, which often improves accuracy and
consistency of format. LangChain provides two builders:

    FewShotPromptTemplate            -> for plain-text (legacy LLM) prompts
    FewShotChatMessagePromptTemplate -> for chat-message prompts, rendering
                                         each example as a human/ai turn pair

Both take a list of example dicts and a template describing how to format
each example.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_community.chat_models import ChatOllama

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

# --- FewShotPromptTemplate: plain text -------------------------------------
example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

few_shot_text_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"],
)

print(few_shot_text_prompt.invoke({"word": "big"}).to_string())
# -> "Give the antonym of each word.
#
#     Input: happy
#     Output: sad
#
#     Input: tall
#     Output: short
#
#     Input: fast
#     Output: slow
#
#     Input: big
#     Output:"

# --- FewShotChatMessagePromptTemplate: message pairs -----------------------
example_chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)

few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_chat_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the antonym of each word the user provides."),
        few_shot_chat_prompt,  # expands into 3 human/ai message pairs
        ("human", "{word}"),
    ]
)

print(final_prompt.invoke({"word": "big"}).to_messages())
# -> [SystemMessage(content='Give the antonym of each word the user provides.'),
#     HumanMessage(content='happy'), AIMessage(content='sad'),
#     HumanMessage(content='tall'), AIMessage(content='short'),
#     HumanMessage(content='fast'), AIMessage(content='slow'),
#     HumanMessage(content='big')]

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
chain = final_prompt | model

response = chain.invoke({"word": "big"})
print(response.content)
# -> "small"
