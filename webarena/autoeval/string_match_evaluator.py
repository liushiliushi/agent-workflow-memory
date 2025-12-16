"""
LLM-based evaluator for string_match tasks.

This evaluator uses an LLM to determine if the agent's response
correctly answers the question, instead of using strict exact_match.
"""

import re
import json
from typing import Dict, Optional, Tuple
from .clients import LM_Client, OpenRouter_Client


def build_string_match_eval_prompt(intent: str, response: str, reference_answer: str) -> Tuple[str, str]:
    """
    Build prompt for LLM to evaluate if agent's response correctly answers the question.

    Args:
        intent: The user's question/intent
        response: The agent's response
        reference_answer: The expected answer

    Returns:
        Tuple of (prompt, system_msg)
    """
    system_msg = """You are an expert in evaluating the correctness of agent responses to user questions.

Given:
1. A user question
2. The agent's response
3. The expected reference answer

Your task is to determine if the agent's response contains the correct answer to the question.

Guidelines:
- The agent's response may be verbose or contain extra explanation - this is OK
- What matters is whether the CORE ANSWER is correct
- Consider variations in formatting, capitalization, punctuation as acceptable
- For numerical answers, check if the numbers match (minor rounding differences are OK)
- For entity names (brands, products, etc.), exact name match is required
- If the agent provides reasoning/explanation along with the correct answer, it's still correct

Format your response as:
Thoughts: <your reasoning about whether the response is correct>
Status: "success" or "failure"
"""

    prompt = f"""User Question: {intent}

Agent's Response: {response}

Expected Answer: {reference_answer}

Does the agent's response correctly answer the user's question?"""

    return prompt, system_msg


class StringMatchEvaluator:
    """LLM-based evaluator for string_match tasks."""

    def __init__(self, model_name: str = "google/gemini-2.5-flash-preview-09-2025", use_openrouter: bool = True):
        """
        Initialize the evaluator.

        Args:
            model_name: The model to use for evaluation
            use_openrouter: Whether to use OpenRouter (for non-OpenAI models)
        """
        self.model_name = model_name
        self.use_openrouter = use_openrouter

        if use_openrouter:
            self.client = OpenRouter_Client(model_name=model_name)
        else:
            self.client = LM_Client(model_name=model_name)

    def evaluate(self, intent: str, agent_response: str, reference_answer: str) -> Dict:
        """
        Evaluate if agent's response correctly answers the question.

        Args:
            intent: The user's question/intent
            agent_response: The agent's response
            reference_answer: The expected answer

        Returns:
            Dict with evaluation results:
            {
                "success": bool,
                "thoughts": str,
                "raw_response": str
            }
        """
        # Build prompt
        prompt, system_msg = build_string_match_eval_prompt(intent, agent_response, reference_answer)

        # Get LLM evaluation
        try:
            response, _ = self.client.one_step_chat(prompt, system_msg=system_msg)

            # Parse response
            thoughts_match = re.search(r'Thoughts:\s*(.+?)(?=Status:|$)', response, re.DOTALL | re.IGNORECASE)
            status_match = re.search(r'Status:\s*["\']?(success|failure)["\']?', response, re.IGNORECASE)

            thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
            status = status_match.group(1).lower() if status_match else "failure"

            return {
                "success": status == "success",
                "thoughts": thoughts,
                "raw_response": response,
                "reward": 1.0 if status == "success" else 0.0
            }

        except Exception as e:
            return {
                "success": False,
                "thoughts": f"Error during evaluation: {str(e)}",
                "raw_response": "",
                "reward": 0.0,
                "error": str(e)
            }

    def evaluate_from_log(self, log_file: str, intent: str, reference_answer: str) -> Optional[Dict]:
        """
        Evaluate from a task log file.

        Args:
            log_file: Path to the task log file
            intent: The user's question/intent
            reference_answer: The expected answer

        Returns:
            Evaluation result dict, or None if no response found
        """
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Find the last send_msg_to_user action
        matches = re.findall(r"send_msg_to_user\(['\"](.+?)['\"]\)", log_content, re.DOTALL)

        if not matches:
            return None

        # Get the last message (final answer)
        agent_response = matches[-1]

        # Evaluate
        return self.evaluate(intent, agent_response, reference_answer)


def reevaluate_task_with_llm(
    task_id: int,
    log_file: Optional[str] = None,
    model_name: str = "google/gemini-2.5-flash-preview-09-2025",
    use_openrouter: bool = True
) -> Dict:
    """
    Re-evaluate a string_match task using LLM.

    Args:
        task_id: The task ID
        log_file: Path to log file (auto-detects if None)
        model_name: Model to use for evaluation
        use_openrouter: Whether to use OpenRouter

    Returns:
        Dict with evaluation results
    """
    import json
    from pathlib import Path

    # Load task config
    config_file = Path(f"config_files/{task_id}.json")
    if not config_file.exists():
        return {"error": f"Config file not found: {config_file}"}

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Check if this task uses string_match evaluation
    eval_config = config.get("eval", {})
    eval_types = eval_config.get("eval_types", [])

    if "string_match" not in eval_types:
        return {"error": "Task does not use string_match evaluation"}

    # Auto-detect log file if not provided
    if log_file is None:
        # Try batch logs first
        batch_dirs = sorted(Path("batch_logs").glob("batch_*"), reverse=True)
        for batch_dir in batch_dirs:
            candidate = batch_dir / f"task_{task_id}.log"
            if candidate.exists():
                log_file = candidate
                break

        # Fall back to results directory
        if log_file is None:
            candidate = Path(f"results/webarena.{task_id}/task_{task_id}.log")
            if candidate.exists():
                log_file = candidate

    if log_file is None:
        return {"error": "Log file not found"}

    # Get reference answer (try all match types)
    reference_answers = eval_config.get("reference_answers", {})
    reference_answer = (
        reference_answers.get("exact_match") or
        reference_answers.get("must_include") or
        reference_answers.get("fuzzy_match") or
        ""
    )

    if not reference_answer:
        return {"error": "No reference answer found in config"}

    # Create evaluator and evaluate
    evaluator = StringMatchEvaluator(model_name=model_name, use_openrouter=use_openrouter)
    intent = config.get("intent", "")

    result = evaluator.evaluate_from_log(str(log_file), intent, reference_answer)

    if result is None:
        return {"error": "Could not extract agent response from log"}

    # Add metadata
    result["task_id"] = task_id
    result["intent"] = intent
    result["reference_answer"] = reference_answer
    result["log_file"] = str(log_file)

    return result


if __name__ == "__main__":
    # Test the evaluator
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate string_match tasks using LLM")
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-preview-09-2025", help="Model to use")
    parser.add_argument("--openrouter", action="store_true", default=True, help="Use OpenRouter")

    args = parser.parse_args()

    result = reevaluate_task_with_llm(args.task, model_name=args.model, use_openrouter=args.openrouter)

    print("\n" + "=" * 60)
    print(f"LLM Evaluation Result for Task {args.task}")
    print("=" * 60)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"Intent: {result['intent']}")
        print(f"Reference Answer: {result['reference_answer']}")
        print(f"Success: {'✅' if result['success'] else '❌'}")
        print(f"Reward: {result['reward']}")
        print(f"\nThoughts: {result['thoughts']}")
