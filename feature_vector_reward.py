from typing import Any


def has_opening_explanation_tag(solution_str: str) -> bool:
    return "<explanation>" in solution_str


def has_closing_explanation_tag(solution_str: str) -> bool:
    return "</explanation>" in solution_str


def parse_explanation(solution_str: str) -> str | None:
    if has_opening_explanation_tag(solution_str) and has_closing_explanation_tag(solution_str):
        return solution_str.split("<explanation>")[1].split("</explanation>")[0]
    return None


def compute_score(
    data_source, solution_str: str, ground_truth, extra_info: dict[str, Any]
) -> dict[str, str | float | bool | None]:
    """
    Custom reward function for math problems using proper verl interface.

    Args:
        data_source: The input data/prompt (not used in this implementation)
        solution_str: The model's generated response/solution
        ground_truth: The expected answer
        extra_info: Dict containing batch information, expects 'all_lengths' key with list of lengths

    Returns:
        Float reward score for this single example
    """
    total_reward = 0.0

    if has_opening_explanation_tag(solution_str):
        # partial reward
        total_reward += 0.25
    if has_closing_explanation_tag(solution_str):
        # partial reward
        total_reward += 0.25

    parsed_answer = parse_explanation(solution_str)

    return {
        "score": total_reward,
        "parsed_answer": parsed_answer,
    }
