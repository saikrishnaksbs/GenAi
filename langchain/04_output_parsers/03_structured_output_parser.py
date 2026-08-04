"""
STRUCTURED OUTPUT PARSER
=========================
`StructuredOutputParser` is a lighter-weight alternative to
`PydanticOutputParser` for when you don't want to define a full Pydantic
model. You describe each output field with a `ResponseSchema` (name +
description), and the parser builds format instructions and parses the
model's JSON response into a plain dict of those fields.

It's handy for quick prototypes or when the shape of the data is simple
and doesn't need type validation beyond "these keys should exist."
"""

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Each ResponseSchema describes one key in the expected JSON output.
response_schemas = [
    ResponseSchema(name="sentiment", description="either 'positive', 'negative', or 'neutral'"),
    ResponseSchema(name="summary", description="a one-sentence summary of the review"),
    ResponseSchema(name="rating_guess", description="a guessed star rating from 1 to 5, as an integer"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# The parser generates instructions telling the model exactly how to
# format its JSON response, including the expected keys.
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You analyze product reviews.\n{format_instructions}"),
        ("human", "{review}"),
    ]
).partial(format_instructions=format_instructions)

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
chain = prompt | model | parser

review = "This blender broke after two uses. Total waste of money."
result = chain.invoke({"review": review})

print(result)
# -> {
#      "sentiment": "negative",
#      "summary": "The blender stopped working shortly after purchase.",
#      "rating_guess": 1,
#    }
print(type(result))
# -> <class 'dict'>
print(result["sentiment"])
# -> "negative"

# Because the output is a plain dict, downstream Runnables can pull
# individual fields out with a lambda or RunnableLambda instead of
# needing attribute access.
from langchain_core.runnables import RunnableLambda

extract_summary = RunnableLambda(lambda parsed: parsed["summary"])
summary_only_chain = chain | extract_summary
print(summary_only_chain.invoke({"review": review}))
# -> "The blender stopped working shortly after purchase."
