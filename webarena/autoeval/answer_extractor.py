"""
Answer Extractor for WebArena Tasks

This module extracts concise answers from verbose Agent responses
to improve evaluation accuracy with exact_match evaluators.
"""

import re
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class AnswerExtractor:
    """Extract concise answers from verbose agent responses."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, use_llm: bool = True):
        """
        Initialize the answer extractor.

        Args:
            model_name: The model to use for extraction (default: gpt-4o-mini for cost efficiency)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            use_llm: Whether to use LLM fallback (default: True, set to False for rule-based only)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.use_llm = use_llm

        if use_llm:
            if not self.api_key:
                print("Warning: OpenAI API key not found. Falling back to rule-based extraction only.")
                self.use_llm = False
                self.llm = None
            else:
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0.0,  # Deterministic extraction
                    openai_api_key=self.api_key,
                )
        else:
            self.llm = None

    def extract_answer(self, agent_response: str, question: str) -> str:
        """
        Extract the core answer from a verbose agent response.

        Args:
            agent_response: The verbose response from the agent
            question: The original question/intent

        Returns:
            The extracted concise answer
        """
        # First try rule-based extraction for common patterns
        rule_based_answer = self._rule_based_extraction(agent_response, question)
        if rule_based_answer:
            return rule_based_answer

        # Fall back to LLM-based extraction (if available)
        if self.use_llm and self.llm is not None:
            return self._llm_based_extraction(agent_response, question)

        # If no LLM available and rule-based failed, return the response as-is
        # (This will likely fail exact_match but better than crashing)
        return agent_response

    def _rule_based_extraction(self, agent_response: str, question: str) -> Optional[str]:
        """
        Try to extract answer using simple rules.

        Common patterns:
        - "the answer is X"
        - "the top-1 ... is X"
        - "the best-selling ... is X"
        """
        # Pattern 1: "the answer is X" or "answer: X"
        match = re.search(r'(?:the\s+)?answer(?:\s+is)?[:：]\s*([^,\.]+)', agent_response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: "the top-N ... is X" (improved to handle "brand ... is X")
        match = re.search(r'(?:top-\d+|best-selling)\s+(?:brand|product|item|category|seller)[^.]*?\bis\s+([A-Z][a-zA-Z0-9\s-]*?)(?:\s*,|\s+with|\s*\.)', agent_response)
        if match:
            return match.group(1).strip()

        # Pattern 3: "... is X, with ..." (common in detailed explanations)
        match = re.search(r'\bis\s+([A-Z][a-zA-Z0-9\s-]+?)\s*,\s+with', agent_response)
        if match:
            return match.group(1).strip()

        # Pattern 4: Look for capitalized brand/entity names after "is"
        match = re.search(r'\bis\s+([A-Z][a-zA-Z0-9]+)(?:\s*,|\s*\.|\s+with)', agent_response)
        if match:
            return match.group(1).strip()

        # Pattern 5: Look for quoted text that might be the answer
        quoted_matches = re.findall(r'["""\'](.*?)["""\']', agent_response)
        if len(quoted_matches) == 1:
            return quoted_matches[0].strip()

        return None

    def _llm_based_extraction(self, agent_response: str, question: str) -> str:
        """
        Use LLM to extract the concise answer.

        Args:
            agent_response: The verbose response from the agent
            question: The original question

        Returns:
            The extracted concise answer
        """
        system_prompt = """You are an answer extractor. Your job is to extract the CORE ANSWER from verbose agent responses.

Rules:
1. Extract ONLY the direct answer to the question, nothing else
2. Remove all explanations, reasoning, and context
3. If the answer is a brand name, product name, or entity, return just that name
4. If the answer is a number, return just the number
5. Do NOT add any explanations or additional text
6. If there are multiple possible answers, choose the most prominent one

Examples:
Q: What is the top-1 best-selling brand?
Agent: "Based on the Bestsellers Report for Q1 2022, the top-1 best-selling brand is Sprite, with 4 units sold."
Extracted: Sprite

Q: What is the price of Product X?
Agent: "I found that Product X costs $49.99 according to the product page."
Extracted: $49.99

Q: Which category has the most items?
Agent: "After analyzing the data, Electronics has 152 items, which is the highest."
Extracted: Electronics
"""

        user_prompt = f"""Question: {question}

Agent Response: {agent_response}

Extract the core answer (only the answer, nothing else):"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()


def extract_answer_from_log(log_file: str, question: str, extractor: Optional[AnswerExtractor] = None) -> Optional[str]:
    """
    Extract answer from a task log file.

    Args:
        log_file: Path to the task log file
        question: The original question/intent
        extractor: AnswerExtractor instance (creates new one if None)

    Returns:
        The extracted answer, or None if no answer found
    """
    if extractor is None:
        extractor = AnswerExtractor()

    with open(log_file, 'r') as f:
        log_content = f.read()

    # Find the last send_msg_to_user action
    # Pattern: send_msg_to_user('...')
    matches = re.findall(r"send_msg_to_user\(['\"](.+?)['\"]\)", log_content, re.DOTALL)

    if not matches:
        return None

    # Get the last message (final answer)
    agent_response = matches[-1]

    # Extract the concise answer
    return extractor.extract_answer(agent_response, question)


if __name__ == "__main__":
    # Test the extractor
    extractor = AnswerExtractor()

    # Test case 1: Task 1
    question = "What is the top-1 best-selling brand in Quarter 1 2022"
    agent_response = "Based on the Bestsellers Report for Q1 2022 (Jan 1, 2022 to Mar 31, 2022), the top-1 best-selling brand by order quantity is Sprite, with 4 units sold (Sprite Yoga Strap 8 foot: 2 units, Sprite Yoga Strap 6 foot: 2 units)."

    print("Test Case 1:")
    print(f"Question: {question}")
    print(f"Agent Response: {agent_response}")
    print(f"Extracted Answer: {extractor.extract_answer(agent_response, question)}")
    print()

    # Test case 2: Generic pattern
    question = "What is the price of the product?"
    agent_response = "After checking the product page, I found that the product costs $29.99 including tax."

    print("Test Case 2:")
    print(f"Question: {question}")
    print(f"Agent Response: {agent_response}")
    print(f"Extracted Answer: {extractor.extract_answer(agent_response, question)}")
