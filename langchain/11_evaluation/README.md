# LangChain Evaluation

This directory details LangChain's evaluation framework (`langchain.evaluation`), which allows you to audit, grade, and compare model completions. It covers string-level reference matches, criteria assessments, pairwise comparison grading, and agent execution path evaluations.

---

## Table of Contents
1. [Overview of Evaluators](#1-overview-of-evaluators)
2. [String Evaluators](#2-string-evaluators)
3. [Criteria Evaluators](#3-criteria-evaluators)
4. [Pairwise Comparison Evaluators (A/B Testing)](#4-pairwise-comparison-evaluators-ab-testing)
5. [Trajectory Evaluators (Agent Auditing)](#5-trajectory-evaluators-agent-auditing)
6. [Unit Testing with Mocked LLM Responses](#6-unit-testing-with-mocked-llm-responses)
7. [Mocking Tool Calls for Agents](#7-mocking-tool-calls-for-agents)
8. [Prompt Regression Testing](#8-prompt-regression-testing)
9. [Production A/B Testing on Live Traffic](#9-production-ab-testing-on-live-traffic)

---

## 1. Overview of Evaluators
To verify the performance of LLM chains, LangChain provides a set of evaluators. These evaluators run structured judging prompts (often powered by a secondary "judge" model) to analyze completions, assign scores, and output structured reasoning verdicts.

---

## 2. String Evaluators
[String Evaluators](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/01_string_and_criteria_evaluators.py#L4-L8) compare a model's prediction against a known ground-truth reference answer.
- Uses semantic embedding similarities or LLM judges to grade the match, returning scores (typically 0 or 1) and supporting reasoning details.
- Useful for regression testing when you have a pre-defined evaluation dataset.

---

## 3. Criteria Evaluators
Criteria Evaluators assess a completion against a specific guideline without requiring a reference answer (details in [01_string_and_criteria_evaluators.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/01_string_and_criteria_evaluators.py#L9-L12)).
- LangChain includes pre-configured criteria such as **`conciseness`**, **`correctness`**, **`coherence`**, and **`harmfulness`**.
- You can also write custom criteria by specifying a description of the desired quality constraint.
- The judge LLM processes the completion, outputs a structured breakdown of how the text met or failed the criteria, and returns a binary pass/fail score.

---

## 4. Pairwise Comparison Evaluators (A/B Testing)
[Pairwise Evaluators](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/02_pairwise_comparison_evaluators.py#L4-L9) are used for comparing two different completions (e.g. from two different models, prompts, or chain iterations) for the same input.
- A judge LLM compares the two predictions side-by-side and selects the superior output.
- **Position Bias**: LLM judges tend to favor whichever answer is presented first in their context. To mitigate this, run the comparison twice with the candidate order swapped, validating that the preference remains consistent.

---

## 5. Trajectory Evaluators (Agent Auditing)
While output-only evaluators only check the final response, [Trajectory Evaluators](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/03_trajectory_evaluators.py#L4-L9) audit the agent's step-by-step path.
- The evaluator reviews the full sequence of actions (`AgentAction` events) and observations (`ToolMessage` payloads) that occurred during execution.
- Evaluates:
  - Did the agent call appropriate tools?
  - Was the execution order logical?
  - Did the agent get stuck in redundant loops?
  - Did the agent attempt any forbidden actions?
- This is critical for evaluating agent safety and efficiency.

---

## 6. Unit Testing with Mocked LLM Responses
Unit tests must verify output parsing, prompt formatting, and error handling without making expensive and slow calls to external API endpoints.
- **`FakeMessagesListChatModel`**: Simulates model interaction by sequentially returning predetermined `AIMessage` inputs.
- **`unittest.mock`**: Mocks the internal `invoke` / `stream` call of your wrapper and asserts that the formatted prompt structure conforms to expectations.

See [04_mock_llm_testing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/04_mock_llm_testing.py) for examples.

---

## 7. Mocking Tool Calls for Agents
Testing an agent loop locally without triggering DB writes or API charges.
- **Tool Mocking**: Mocking the `_run` method of a custom tool to verify that the agent supplies the correct arguments.
- **Loop Orchestration Testing**: Injected mocks for tool responses inside a mock trace runner.

See [05_mock_tool_testing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/05_mock_tool_testing.py) for examples.

---

## 8. Prompt Regression Testing
Ensuring prompt changes designed for new features do not break previously correct answers.
- **Golden Dataset**: Maintain a dictionary of queries, expected keywords, and forbidden assertions.
- **Automated Validation**: Compare updated prompt outputs against constraints to prevent breaking changes in production.

See [06_regression_testing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/06_regression_testing.py) for details.

---

## 9. Production A/B Testing on Live Traffic
Testing prompt performance live in production environments.
- **Traffic Routing**: Randomizing or session-pinning user queries between control (Prompt A) and treatment (Prompt B) paths.
- **Trace Tagging**: Applying metadata tags to traces to filter and compare performance metrics in platforms like LangSmith.

See [07_live_ab_testing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/11_evaluation/07_live_ab_testing.py) for details.
