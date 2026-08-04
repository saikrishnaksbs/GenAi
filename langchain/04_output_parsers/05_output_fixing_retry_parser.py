"""
OUTPUT FIXING AND RETRY PARSERS
=================================
Real models occasionally produce malformed output that a strict parser
(like PydanticOutputParser) will reject with an exception. Rather than
crashing your pipeline, you can wrap the parser in a self-healing layer:

  - `OutputFixingParser` catches a parsing failure and asks a model to
    repair the broken text so it conforms to the schema, then re-parses.
  - `RetryOutputParser` goes further: it also has access to the original
    prompt, so it can re-generate a proper response from scratch rather
    than just patching malformed text.

Both wrap an existing parser and are drop-in replacements for it.
"""

from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="the person's full name")
    age: int = Field(description="the person's age in years")


base_parser = PydanticOutputParser(pydantic_object=Person)
fixing_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# Suppose the raw model output was malformed (missing quotes / trailing comma):
malformed_output = "{name: 'Ada Lovelace', age: 36,}"

try:
    base_parser.parse(malformed_output)
except Exception as exc:
    print(f"base parser failed: {exc}")
    # -> base parser failed: Invalid json output: {name: 'Ada Lovelace', age: 36,}

# --- OutputFixingParser -----------------------------------------------
# Wraps base_parser; on failure it sends the bad output + error back to
# an LLM with instructions to fix it, then retries parsing the result.
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=fixing_model)

fixed_person: Person = fixing_parser.parse(malformed_output)
print(fixed_person)
# -> name='Ada Lovelace' age=36

# --- RetryOutputParser ---------------------------------------------------
# Needs the original prompt (as a PromptValue) in addition to the bad
# completion, because it may choose to regenerate the answer entirely
# rather than just patch the text.
retry_parser = RetryOutputParser.from_llm(parser=base_parser, llm=fixing_model)

original_prompt = StringPromptValue(
    text="Extract the person's name and age as JSON: 'Ada Lovelace is 36 years old.'"
)

retried_person: Person = retry_parser.parse_with_prompt(malformed_output, original_prompt)
print(retried_person)
# -> name='Ada Lovelace' age=36

# In an LCEL chain, you'd typically catch the parse failure and fall back
# to the fixing parser rather than letting the chain raise:
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Extract the person's name and age as JSON.\n{format_instructions}\n{text}",
    input_variables=["text"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()},
)
resilient_chain = prompt | fixing_model | fixing_parser  # fixing_parser replaces base_parser

result = resilient_chain.invoke({"text": "Grace Hopper, age 85, was a computer scientist."})
print(result)
# -> name='Grace Hopper' age=85
