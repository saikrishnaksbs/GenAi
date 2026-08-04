"""
REGRESSION TESTING FOR PROMPT MODIFICATIONS
============================================
Prompts are code. When you optimize a prompt for a new feature or edge case, 
you risk introducing regressions for previously working inputs.

This script demonstrates how to set up an offline prompt regression suite:
1. Define a "golden dataset" (inputs and expected assertion checks).
2. Run prompt iterations through the model.
3. Automatically grade outputs using string criteria and assertions.
"""

import unittest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Golden dataset: test queries and the rules they must satisfy in the output
GOLDEN_DATASET = [
    {
        "query": "Can I return a defective item after 45 days?",
        "assertion_keywords": ["defective", "warranty", "refund", "yes"],
        "forbidden_keywords": ["strict 30-day limit"]
    },
    {
        "query": "I changed my mind. The item is fine. Can I return it after 45 days?",
        "assertion_keywords": ["30 days", "no", "cannot return"],
        "forbidden_keywords": ["defective", "warranty"]
    }
]

# --- Prompt version A (Baseline version) ---
baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are customer service. Our return window is 30 days. "
               "However, defective items are covered under warranty for 90 days."),
    ("human", "{query}")
])

# --- Prompt version B (Modified/Optimized prompt that might cause regression if buggy) ---
buggy_modified_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are customer service. We do not accept returns after 30 days under any circumstances."),
    ("human", "{query}")
])


class TestPromptRegression(unittest.TestCase):
    
    def setUp(self):
        self.model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

    def test_baseline_prompt_passes_golden_set(self):
        """Verify baseline prompt meets all golden criteria."""
        chain = baseline_prompt | self.model | StrOutputParser()
        
        failures = []
        for case in GOLDEN_DATASET:
            response = chain.invoke({"query": case["query"]})
            content = response.lower()
            
            # Check assertions
            for kw in case["assertion_keywords"]:
                if kw.lower() not in content:
                    failures.append(f"Query: '{case['query']}'. Expected keyword '{kw}' missing. Output: {response}")
            
            # Check forbidden constraints
            for kw in case["forbidden_keywords"]:
                if kw.lower() in content:
                    failures.append(f"Query: '{case['query']}'. Forbidden keyword '{kw}' present. Output: {response}")
                    
        # Assert no failures happened
        self.assertEqual(len(failures), 0, "\n".join(failures))

    def test_modified_prompt_regression_check(self):
        """This test verifies if the updated prompt regresses on warranty rules."""
        chain = buggy_modified_prompt | self.model | StrOutputParser()
        
        failures = []
        for case in GOLDEN_DATASET:
            response = chain.invoke({"query": case["query"]})
            content = response.lower()
            
            for kw in case["assertion_keywords"]:
                if kw.lower() not in content:
                    failures.append(f"Query: '{case['query']}'. Expected keyword '{kw}' missing. Output: {response}")
            
            for kw in case["forbidden_keywords"]:
                if kw.lower() in content:
                    failures.append(f"Query: '{case['query']}'. Forbidden keyword '{kw}' present. Output: {response}")
        
        # We expect this test to FAIL (raising AssertionErrors) because the buggy modified prompt
        # forgets about the 90-day warranty rule, triggering a regression!
        if len(failures) > 0:
            print("\n[REGRESSION DETECTED] Modified prompt failed regression checks:")
            print("\n".join(failures))
            # In a CI pipeline, this assertion fails the build:
            # self.fail("Regression detected on updated prompt!")


if __name__ == "__main__":
    unittest.main()
