"""
PipelinePromptTemplate
=========================
PipelinePromptTemplate lets you build a "final" prompt out of several
reusable sub-prompts (e.g. a shared persona block + a task-specific block +
an examples block). Each sub-prompt is rendered independently and its
output is injected as a variable into the final template.

NOTE: In newer LangChain versions this pattern is often replaced by simply
composing PromptTemplates with `+` (see 06_partial_prompts_and_composition.py),
but PipelinePromptTemplate is still useful for deeply nested, reusable blocks.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.pipeline import PipelinePromptTemplate

# The final template references variables that will be filled in by sub-prompts.
full_template = PromptTemplate.from_template(
    "{persona}\n\n{task}\n\n{examples}"
)

persona_prompt = PromptTemplate.from_template(
    "You are {name}, an expert {role}."
)

task_prompt = PromptTemplate.from_template(
    "Task: {task_description}"
)

examples_prompt = PromptTemplate.from_template(
    "Examples:\n{example_list}"
)

# Each tuple maps a variable name in `full_template` to the sub-prompt that fills it.
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("persona", persona_prompt),
        ("task", task_prompt),
        ("examples", examples_prompt),
    ],
)

print(pipeline_prompt.input_variables)
# -> ['name', 'role', 'task_description', 'example_list']

result = pipeline_prompt.format(
    name="Ada",
    role="Python tutor",
    task_description="Explain closures",
    example_list="1. counter functions\n2. decorators",
)
print(result)
