"""
01. Model Initialization
========================
This script demonstrates how to initialize different types of Chat Models in LangChain.
LangChain provides a unified interface for interacting with various LLM providers.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

# Load environment variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
load_dotenv()

def main():
    print("--- 1. Initializing OpenAI ---")
    # Requires OPENAI_API_KEY in environment variables
    # We set temperature=0 for deterministic outputs
    try:
        openai_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = openai_model.invoke("Hello, how are you?")
        print("OpenAI Response:", response.content)
    except Exception as e:
        print("Could not run OpenAI model (missing key?):", e)
        
    print("\n--- 2. Initializing Anthropic ---")
    # Requires ANTHROPIC_API_KEY in environment variables
    try:
        anthropic_model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        response = anthropic_model.invoke("Hello, how are you?")
        print("Anthropic Response:", response.content)
    except Exception as e:
        print("Could not run Anthropic model (missing key?):", e)
        
    print("\n--- 3. Initializing Local Model (Ollama) ---")
    # Requires Ollama running locally with the qwen2.5:1.5b model (or change to your model)
    try:
        ollama_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
        response = ollama_model.invoke("Hello, how are you?")
        print("Ollama Response:", response.content)
    except Exception as e:
        print("Could not run Ollama model (is it running?):", e)

if __name__ == "__main__":
    main()
