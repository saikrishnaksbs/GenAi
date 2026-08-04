"""
MOCKING TOOL CALLS FOR AGENT TESTING
====================================
Agents solve complex workflows by calling tools. Testing agents in isolation 
requires mocking these tools so they don't hit external databases, trigger 
emails, or charge credit cards.

This script demonstrates how to:
1. Define a tool using LangChain.
2. Mock the tool's execution under a mock test harness.
3. Test agent responses with simulated tool outputs.
"""

import unittest
from unittest.mock import MagicMock
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


# 1. Define a tool that we want to test/mock
@tool
def check_order_status(order_id: str) -> str:
    """Checks the delivery status of an order in the remote database."""
    # In a real environment, this would call database/API.
    return "Status: Shipped via DHL, ETA Tomorrow."


# 2. Write a mock agent test harness
class TestAgentToolMocking(unittest.TestCase):
    
    def test_mock_tool_execution(self):
        """Verify we can mock the tool logic directly."""
        # Save a reference to the original function
        original_func = check_order_status._run
        
        try:
            # Mock the internal execution function of the structured tool
            check_order_status._run = MagicMock(return_value="Status: Cancelled")
            
            # Invoke the tool
            result = check_order_status.invoke({"order_id": "12345"})
            
            # Assert mock was hit with correct arguments and mock output returned
            check_order_status._run.assert_called_once_with("12345")
            self.assertEqual(result, "Status: Cancelled")
            
        finally:
            # Always restore original state to prevent test contamination
            check_order_status._run = original_func

    def test_agent_tool_calling_flow_with_mock_llm(self):
        """Simulate an entire agent reasoning loop using FakeMessagesListChatModel.
        
        Step 1: LLM returns tool call request.
        Step 2: Runner calls the mock tool.
        Step 3: LLM receives tool output and finishes.
        """
        # Step 1: Model requests a tool call to 'check_order_status'
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "check_order_status",
                "args": {"order_id": "9999"},
                "id": "call_abc123"
            }]
        )
        
        # Step 3: Model summarizes after tool output is injected
        final_msg = AIMessage(content="Order 9999 is shipped and will arrive tomorrow.")
        
        # Setup Fake LLM sequence
        fake_llm = FakeMessagesListChatModel(responses=[tool_call_msg, final_msg])
        
        # Simple Agent run loop simulation
        query = "Where is my order 9999?"
        messages = [AIMessage(content=query)]
        
        # 1. First model call (asks for tool)
        response1 = fake_llm.invoke(messages)
        messages.append(response1)
        
        self.assertTrue(len(response1.tool_calls) > 0)
        tool_call = response1.tool_calls[0]
        
        # 2. execute tool (mocking the execution to return a specific state)
        mock_output = "Status: Delighted mock delivery complete."
        
        # Create the ToolMessage containing mock results
        tool_msg = ToolMessage(
            content=mock_output,
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        messages.append(tool_msg)
        
        # 3. Second model call (receives tool output, returns answer)
        response2 = fake_llm.invoke(messages)
        
        self.assertEqual(response2.content, "Order 9999 is shipped and will arrive tomorrow.")
        self.assertEqual(len(messages), 3) # Human -> AI (Tool call request) -> Tool response


if __name__ == "__main__":
    unittest.main()
