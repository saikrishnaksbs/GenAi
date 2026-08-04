"""
LIST AND XML OUTPUT PARSERS
=============================
`CommaSeparatedListOutputParser` splits a comma-separated string response
into a Python list — useful for "give me N items" style prompts without
needing full JSON structure.

`XMLOutputParser` parses model output written in XML tags into a nested
dict, which is useful for models (like some Anthropic models) that tend
to produce well-structured XML more reliably than JSON.
"""

from langchain_core.output_parsers import CommaSeparatedListOutputParser, XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- CommaSeparatedListOutputParser -----------------------------------
list_parser = CommaSeparatedListOutputParser()

list_prompt = PromptTemplate(
    template="List 5 {category}.\n{format_instructions}",
    input_variables=["category"],
    partial_variables={"format_instructions": list_parser.get_format_instructions()},
)
# -> format_instructions look like:
#    "Your response should be a list of comma separated values, eg: `foo, bar, baz`"

list_chain = list_prompt | model | list_parser
fruits = list_chain.invoke({"category": "tropical fruits"})

print(fruits)
# -> ["mango", "papaya", "pineapple", "guava", "dragon fruit"]
print(type(fruits))
# -> <class 'list'>
print(len(fruits))
# -> 5

# --- XMLOutputParser -----------------------------------------------------
from langchain_community.chat_models import ChatOllama

claude = ChatOllama(model="qwen2.5:1.5b", temperature=0)
xml_parser = XMLOutputParser(tags=["movie", "title", "year", "genre"])

xml_prompt = PromptTemplate(
    template="Describe the movie {title} using this XML format.\n{format_instructions}",
    input_variables=["title"],
    partial_variables={"format_instructions": xml_parser.get_format_instructions()},
)

xml_chain = xml_prompt | claude | xml_parser
parsed = xml_chain.invoke({"title": "Inception"})

print(parsed)
# -> {"movie": [{"title": "Inception"}, {"year": "2010"}, {"genre": "Sci-Fi"}]}

# The expected raw model output before parsing would look like:
#   <movie>
#     <title>Inception</title>
#     <year>2010</year>
#     <genre>Sci-Fi</genre>
#   </movie>

# XMLOutputParser also supports streaming: it yields incrementally
# deeper partial dicts as more closing tags are seen.
for partial in xml_chain.stream({"title": "Inception"}):
    print(partial)
# -> {"movie": [{"title": "Inception"}]}  {"movie": [{"title": "Inception"}, {"year": "2010"}]}  ...
