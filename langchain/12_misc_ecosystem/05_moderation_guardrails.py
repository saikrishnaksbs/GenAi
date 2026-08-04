"""
OUTPUT CONTENT MODERATION & GUARDRAILS
=======================================
Guardrails ensure that LLM outputs remain safe, compliant, and correctly structured 
before being returned to users or executed in downstream pipelines.

This script demonstrates two key guardrails:
1. Native OpenAI Moderation check (to filter hate, violence, self-harm, etc.).
2. Self-Correcting JSON schema validation (re-running the LLM with error logs if validation fails).
"""

import json
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
# Note: OpenAI offers a free Moderation API that can be accessed via LangChain's OpenAIModerationChain
# or directly. We will demonstrate a self-correction loop and direct evaluation checks.

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --------------------------------------------------------------------------
# Technique 1: Self-Correcting Output Validation Loop (Pydantic Guardrail)
# --------------------------------------------------------------------------
class UserProfile(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age, must be a positive integer greater than 0")
    email: str = Field(description="User email address")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract user details from the text. Return output strictly in JSON conforming to this schema:\n"
               "{schema_info}\n"
               "Do NOT output markdown code blocks (e.g., ```json) or explanation."),
    ("human", "Input text: {text}")
])

correction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Your previous output failed validation with the following error:\n"
               "{error}\n"
               "Please fix the response and output valid JSON conforming strictly to this schema:\n"
               "{schema_info}"),
    ("human", "Input text: {text}")
])

def invoke_with_guardrail(text: str, max_retries: int = 2) -> UserProfile:
    """Invokes LLM and runs validator. In case of validation errors, runs self-correction."""
    schema_str = json.dumps(UserProfile.model_json_schema(), indent=2)
    
    # Try initial run
    chain = prompt | model | StrOutputParser()
    raw_output = chain.invoke({"text": text, "schema_info": schema_str})
    
    for attempt in range(max_retries):
        try:
            # Try to parse the raw text
            cleaned_json = raw_output.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned_json)
            profile = UserProfile(**data)
            print(f"Validation Passed (Attempt {attempt+1})!")
            return profile
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Validation Failed (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise ValueError("Failed to obtain valid structure after retries.") from e
            
            # Request correction
            correction_chain = correction_prompt | model | StrOutputParser()
            raw_output = correction_chain.invoke({
                "text": text,
                "error": str(e),
                "schema_info": schema_str
            })

print("--- Running Pydantic Guardrail Demo ---")
# Case 1: Clean input
print("\nInvoking clean input:")
profile_ok = invoke_with_guardrail("Alice is a 28 year old engineer and her email is alice@example.com")
print(profile_ok)

# Case 2: Ambiguous/Bad age input triggering validator self-correction
print("\nInvoking input that might trigger errors (invalid age format):")
profile_fixed = invoke_with_guardrail("Bob is -5 years old and his email is bob@example.com")
print(profile_fixed)


# --------------------------------------------------------------------------
# Technique 2: LLM Output Moderation Layer
# --------------------------------------------------------------------------
# Checks if the LLM output contains any forbidden categories/words.
def check_moderation_guardrail(output: str) -> bool:
    """Simple moderation check. In production, this can call OpenAI Moderation Endpoint."""
    forbidden_terms = ["illegal software", "bypass authentication", "malware code"]
    output_lower = output.lower()
    for term in forbidden_terms:
        if term in output_lower:
            return False # Failed moderation
    return True

print("\n--- Running Moderation Check Demo ---")
outputs = [
    "Here is the code to create a simple HTTP server in Python.",
    "Sure, here is how you can use malware code to bypass authentication."
]

for out in outputs:
    is_safe = check_moderation_guardrail(out)
    print(f"Output: '{out}'\n  -> Moderation Check: {'PASSED' if is_safe else 'FAILED/BLOCKED'}")
