"""
PYDANTIC / JSON OUTPUT PARSER
==============================
`PydanticOutputParser` coerces a model's text response into a validated
Pydantic object, giving you typed, structured data instead of a free-form
string. `JsonOutputParser` is the looser cousin: it parses to a plain dict
and can optionally validate against a Pydantic schema too.

Both parsers expose `.get_format_instructions()`, a string you inject into
the prompt so the model knows exactly what JSON shape to produce.
"""

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)


class Recipe(BaseModel):
    name: str = Field(description="the name of the dish")
    ingredients: list[str] = Field(description="list of ingredients needed")
    minutes_to_cook: int = Field(description="estimated cooking time in minutes")


pydantic_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = PromptTemplate(
    template="Suggest a simple recipe for {dish}.\n{format_instructions}\n",
    input_variables=["dish"],
    # Bakes JSON-schema instructions straight into the prompt text.
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)

chain = prompt | model | pydantic_parser
recipe: Recipe = chain.invoke({"dish": "banana pancakes"})

print(recipe.name)
# -> "Banana Pancakes"
print(recipe.ingredients)
# -> ["bananas", "flour", "eggs", "milk", "baking powder"]
print(recipe.minutes_to_cook)
# -> 15
print(type(recipe))
# -> <class '__main__.Recipe'>

# JsonOutputParser is a lighter-weight alternative that returns a plain
# dict rather than a validated model instance, while still accepting a
# Pydantic schema to steer the format instructions.
json_parser = JsonOutputParser(pydantic_object=Recipe)
json_prompt = PromptTemplate(
    template="Suggest a simple recipe for {dish}.\n{format_instructions}\n",
    input_variables=["dish"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)
json_chain = json_prompt | model | json_parser

recipe_dict = json_chain.invoke({"dish": "banana pancakes"})
print(recipe_dict)
# -> {"name": "Banana Pancakes", "ingredients": [...], "minutes_to_cook": 15}

# JsonOutputParser also supports streaming partial JSON objects as they
# are generated, which PydanticOutputParser cannot do (it needs the full
# text to validate against the schema).
for partial in json_chain.stream({"dish": "banana pancakes"}):
    print(partial)
# -> {}  {"name": "Ban"}  {"name": "Banana Pancakes", "ingredients": []}  ...
