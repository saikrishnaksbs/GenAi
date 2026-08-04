# LangChain Output Parsers

This directory details LangChain's abstractions for parsing model responses into structured formats (strings, lists, XML, dicts, or validated Pydantic model objects). It also covers self-healing layers that handle parsing failures gracefully.

---

## Table of Contents
1. [StrOutputParser (String Extraction)](#1-stroutputparser-string-extraction)
2. [Pydantic vs. JSON Output Parsers](#2-pydantic-vs-json-output-parsers)
3. [StructuredOutputParser & ResponseSchema](#3-structuredoutputparser--responseschema)
4. [List & XML Parsers](#4-list--xml-parsers)
5. [Self-Healing Parsers: OutputFixing & Retry](#5-self-healing-parsers-outputfixing--retry)

---

## 1. StrOutputParser (String Extraction)
[StrOutputParser](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/01_str_output_parser.py#L4-L10) is the simplest and most commonly used parser in LangChain. It takes the returned `AIMessage` and extracts the `.content` string, discarding metadata like usage stats and token configurations.

In [01_str_output_parser.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/01_str_output_parser.py):
```python
from langchain_core.output_parsers import StrOutputParser

# StrOutputParser fits at the end of the LCEL chain
chain = prompt | model | StrOutputParser()
```
It supports streaming out-of-the-box, yielding tokens individually as they are received.

---

## 2. Pydantic vs. JSON Output Parsers
For structured data extraction, LangChain offers schema-driven parsers:
- **`PydanticOutputParser`**: Coherses the raw model completion into a validated Pydantic object instance.
- **`JsonOutputParser`**: A lighter parser that yields a standard Python dictionary. It can optionally be passed a Pydantic schema to guide format instructions.

Both parsers expose a `.get_format_instructions()` method that returns a string representation of the JSON schema, which should be injected into the prompt.

Comparison shown in [02_pydantic_json_parser.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/02_pydantic_json_parser.py):
```python
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

class Recipe(BaseModel):
    name: str = Field(description="the name of the dish")
    ingredients: list[str] = Field(...)

pydantic_parser = PydanticOutputParser(pydantic_object=Recipe)
json_parser = JsonOutputParser(pydantic_object=Recipe)
```
- **Streaming Difference**: `JsonOutputParser` supports partial JSON streaming (emitting incomplete dicts as token fragments arrive). `PydanticOutputParser` cannot stream partial states, since full validation requires the entire JSON payload to compile.

---

## 3. StructuredOutputParser & ResponseSchema
[StructuredOutputParser](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/03_structured_output_parser.py#L4-L12) provides a dict-based structuring mechanism without requiring Pydantic class definitions.
- Individual keys are defined using a list of **`ResponseSchema`** objects.
- The parser generates instructions and extracts the keys into a standard dictionary.

In [03_structured_output_parser.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/03_structured_output_parser.py):
```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(name="sentiment", description="positive, negative, or neutral"),
    ResponseSchema(name="rating_guess", description="star rating 1 to 5 as an integer"),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
```

---

## 4. List & XML Parsers

### CommaSeparatedListOutputParser
Splits a comma-separated text string from the model response into a Python list of strings. Ideal for list generations (details in [04_list_and_xml_parsers.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/04_list_and_xml_parsers.py#L20-L21)).
```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
list_parser = CommaSeparatedListOutputParser()
```

### XMLOutputParser
Parses XML tags from the model output into a nested dictionary structure. Highly recommended for models that generate XML formats more reliably than JSON (such as Anthropic Claude).
```python
from langchain_core.output_parsers import XMLOutputParser
xml_parser = XMLOutputParser(tags=["movie", "title", "year"])
```
XMLOutputParser supports streaming by yielding incrementally deeper nested dicts as closing tag elements are completed.

---

## 5. Self-Healing Parsers: OutputFixing & Retry
Strict schemas sometimes fail due to small JSON syntax errors (e.g. missing commas or quotes). LangChain provides self-healing parsers to resolve this:
- **`OutputFixingParser`**: Catches parsing exceptions, packages the malformed text and error trace, and asks an auxiliary LLM model to correct the syntax errors, then parses the corrected output.
- **`RetryOutputParser`**: Takes the malformed text, the error trace, **and the original prompt**. It instructs the model to regenerate the response from scratch, yielding a higher recovery rate for semantic errors.

Demonstrated in [05_output_fixing_retry_parser.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/04_output_parsers/05_output_fixing_retry_parser.py):
```python
from langchain.output_parsers import OutputFixingParser, RetryOutputParser

# Wrap base_parser to auto-repair syntax errors
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=fixing_model)

# Wrap to allow full regeneration with prompt context
retry_parser = RetryOutputParser.from_llm(parser=base_parser, llm=fixing_model)
```
In standard LCEL pipelines, placing `fixing_parser` at the end of the chain will automatically shield downstream execution nodes from parsing exceptions.
