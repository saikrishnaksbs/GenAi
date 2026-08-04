"""
UNIT TESTING CHAINS WITH MOCKED LLM RESPONSES
==============================================
In production systems, unit tests should not hit live LLM provider APIs (e.g. OpenAI)
to avoid cost, latency, flakiness, and dependency requirements.

This script demonstrates how to mock LLM responses using:
1. Native LangChain `FakeMessagesListChatModel` to supply pre-recorded responses.
2. Standard Python `unittest.mock` to assert mock invocations.
"""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_community.chat_models import ChatOllama


# --- Simple Chain to test ---
def create_summary_chain(llm):
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    return prompt | llm | StrOutputParser()


class TestLangChainComponents(unittest.TestCase):
    
    def test_chain_execution_with_fake_llm(self):
        """1. Using FakeMessagesListChatModel to inject pre-configured model responses."""
        # Configure the fake model to return an AIMessage response
        fake_response = AIMessage(content="This is a fake summary output.")
        fake_llm = FakeMessagesListChatModel(responses=[fake_response])
        
        chain = create_summary_chain(fake_llm)
        
        # Invoke the chain
        output = chain.invoke({"text": "A very long document details about some project."})
        
        # Assertions
        self.assertEqual(output, "This is a fake summary output.")

    @patch("langchain_openai.ChatOpenAI.invoke")
    def test_chain_arguments_with_mock(self, mock_invoke):
        """2. Mocking model.invoke using unittest.mock to assert exact argument calls."""
        # Setup mock behavior
        mock_invoke.return_value = AIMessage(content="Mocked model output string.")
        
        real_llm_wrapper = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)
        chain = create_summary_chain(real_llm_wrapper)
        
        # Invoke
        output = chain.invoke({"text": "Hello world"})
        
        # Verify result
        self.assertEqual(output, "Mocked model output string.")
        
        # Assert model.invoke was called with the correctly formatted prompt messages
        mock_invoke.assert_called_once()
        called_args, called_kwargs = mock_invoke.call_args
        
        # Extract the list of messages passed to the model
        messages_list = called_args[0]
        self.assertEqual(len(messages_list), 1)
        self.assertEqual(messages_list[0].content, "Summarize: Hello world")


if __name__ == "__main__":
    unittest.main()
