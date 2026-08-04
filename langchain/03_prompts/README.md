# LangChain Prompts & Templates

This directory details LangChain's prompt template abstractions, focusing on string parameterization, message composition, few-shot demonstration pairs, dynamic runtime example selection, prompt partialing, and pipeline prompt compositions.

---

## Table of Contents
1. [PromptTemplate vs. ChatPromptTemplate](#1-prompttemplate-vs-chatprompttemplate)
2. [MessagesPlaceholder & Conversation Injection](#2-messagesplaceholder--conversation-injection)
3. [Few-Shot Prompting](#3-few-shot-prompting)
4. [Example Selectors & Semantic Similarity](#4-example-selectors--semantic-similarity)
5. [Partial Prompts (Static vs. Lazy/Dynamic Evaluation)](#5-partial-prompts-static-vs-lazydynamic-evaluation)
6. [Prompt Composition (`+` Operator)](#6-prompt-composition--operator)
7. [PipelinePromptTemplate (Nested Prompts)](#7-pipelineprompttemplate-nested-prompts)

---

## 1. PromptTemplate vs. ChatPromptTemplate
LangChain provides two main abstractions for building parameterized prompts:
- **`PromptTemplate`**: Formats a raw, plain-text string. Recommended when working with completion-style LLMs or rendering isolated text strings.
- **`ChatPromptTemplate`**: Formats a list of message objects (`BaseMessage`), constructed from role-template pairs.

In [02_prompt_templates.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/02_prompt_templates.py), templates validation and rendering are shown:
```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Plain string
text_prompt = PromptTemplate.from_template("Write a {length}-sentence summary of {topic}.")
rendered_str = text_prompt.invoke({"length": "two", "topic": "AI"}) # -> PromptValue

# Chat messages
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "Explain {concept}."),
])
rendered_msgs = chat_prompt.invoke({"domain": "science", "concept": "atoms"}) # -> ChatPromptValue
```
Calling `.input_variables` on a template returns the list of placeholder keys that are expected. Invoking a template with missing variables triggers a `KeyError`.

---

## 2. MessagesPlaceholder & Conversation Injection
[MessagesPlaceholder](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/03_messages_placeholder.py#L4-L7) reserves a variable slot inside a `ChatPromptTemplate` to accept a list of message objects dynamically at invocation time. It is commonly used to inject conversation histories or dynamic list variables.

In [03_messages_placeholder.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/03_messages_placeholder.py):
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"), # slot for dynamic messages
    ("human", "{input}"),
])
```
If `optional=True` is set on the placeholder, invoking the template without passing the specified list variable will skip the placeholder instead of throwing a validation error.

---

## 3. Few-Shot Prompting
Few-shot prompting provides in-context example pairs to steer model behavior. LangChain has two few-shot builders:
- **`FewShotPromptTemplate`**: Renders examples as text blocks for completion models.
- **`FewShotChatMessagePromptTemplate`**: Renders examples as message pairs (e.g., Human/AI turns) for chat models.

Example from [04_few_shot_prompts.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/04_few_shot_prompts.py):
```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

examples = [{"input": "happy", "output": "sad"}]
example_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_chat_prompt,
)
```

---

## 4. Example Selectors & Semantic Similarity
When there are too many few-shot examples to fit inside the model's context window, **Example Selectors** choose a subset of examples dynamically based on the input.
- **`LengthBasedExampleSelector`**: Selects as many examples as possible while staying under a specified length budget.
- **`SemanticSimilarityExampleSelector`**: Computes input embeddings and retrieves the top `k` most semantically similar examples using a vector store (e.g., FAISS).
- **`MaxMarginalRelevanceExampleSelector`**: Selects examples based on a balance between semantic similarity and example diversity to avoid redundancy (MMR).

In [05_example_selectors.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/05_example_selectors.py#L58-L63):
```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

similarity_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2
)
```

---

## 5. Partial Prompts (Static vs. Lazy/Dynamic Evaluation)
Partialing binds a subset of template variables to create a new template requiring fewer inputs.
- **Static values**: Pre-filling variables immediately (e.g., standard metadata configuration).
- **Dynamic functions**: Passing a function that is evaluated lazily at formatting time (e.g., retrieving `date.today()` dynamically whenever the prompt is rendered).

In [06_partial_prompts_and_composition.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/06_partial_prompts_and_composition.py):
```python
from datetime import date
template = PromptTemplate.from_template("Today is {today}. {question}")

# Static partial
static_partial = template.partial(today="2026-08-01")

# Lazy partial
dynamic_partial = template.partial(today=lambda: str(date.today()))
```

---

## 6. Prompt Composition (`+` Operator)
Prompt templates can be composed by combining them using the addition operator `+`.
- String prompts concatenate into a single prompt template.
- Chat prompt templates join their message lists together.

Example from [06_partial_prompts_and_composition.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/06_partial_prompts_and_composition.py#L29-L31):
```python
intro = PromptTemplate.from_template("You are {name}.")
task = PromptTemplate.from_template("Your task is {task}.")
combined = intro + "\n" + task
```

---

## 7. PipelinePromptTemplate (Nested Prompts)
[PipelinePromptTemplate](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/03_prompts/07_pipeline_prompt_template.py#L15) lets you construct a complex target prompt from reusable sub-prompts.
- You specify a `final_prompt` and a list of `pipeline_prompts` tuples: `(variable_name, sub_prompt)`.
- The outputs of the sub-prompts are injected into the final prompt's matching placeholders.

```python
from langchain_core.prompts.pipeline import PipelinePromptTemplate

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("persona", persona_prompt),
        ("task", task_prompt),
        ("examples", examples_prompt),
    ]
)
```
While simple prompt addition (`+`) is often preferred for linear concatenation, `PipelinePromptTemplate` remains useful for deeply nested architectures.
