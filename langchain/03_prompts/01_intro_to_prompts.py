"""
01. Intro to Prompts
====================
This script demonstrates the basics of LangChain prompts compared to standard python strings.
Prompt Templates allow for dynamic and reusable ways to generate prompts.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

def main():
    print("--- 1. Standard Python String Formatting (The Old Way) ---")
    # Without LangChain, you'd use f-strings or .format()
    topic = "space exploration"
    raw_prompt = f"Tell me a short joke about {topic}."
    print("Raw Prompt:", raw_prompt)

    print("\n--- 2. LangChain PromptTemplate ---")
    # LangChain's PromptTemplate provides structured templating
    # It automatically infers the input variables
    prompt_template = PromptTemplate.from_template(
        "Tell me a short joke about {topic}."
    )
    
    # We format it by passing the variables
    formatted_prompt = prompt_template.format(topic="artificial intelligence")
    print("Formatted Prompt:", formatted_prompt)
    
    # You can also inspect the variables required by the template
    print("Required variables:", prompt_template.input_variables)

    print("\n--- 3. LangChain ChatPromptTemplate ---")
    # For chat models, it's better to use ChatPromptTemplate which structures
    # the prompt into System, Human, and AI messages.
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that speaks like a {persona}."),
        ("human", "Tell me about {topic}.")
    ])
    
    formatted_chat = chat_template.format_messages(
        persona="pirate",
        topic="the ocean"
    )
    
    print("Formatted Chat Messages:")
    for msg in formatted_chat:
        print(f"  {msg.__class__.__name__}: {msg.content}")

if __name__ == "__main__":
    main()
