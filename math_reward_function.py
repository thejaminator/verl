# ruff: noqa: E722
"""
Math reward function for verl GRPO training.
Based on the reward computation from qwen3_grpo_length_penalty.py and grader.py.

This implements a comprehensive reward system for math problem solving with:
- Format rewards for proper answer tags and structure
- Correctness rewards for accurate mathematical answers
- Length penalty to encourage concise reasoning and penalize overthinking

Total possible reward: 3.5 points
- Basic answer format: 0.5 points (presence of <answer> tags)
- Full format compliance: 0.5 points (complete <think>/<answer> structure)
- Soft format rewards: 1.0 points (partial credit for format elements)
- Correctness: 1.0 points (mathematically correct answer)
- Length penalty: up to ±0.5 points (rewards shorter correct answers, penalizes long incorrect ones)
"""

import regex


def parse_digits(num):
    """Parse numeric strings, handling commas and percentages."""
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    """Check if a string represents a number."""
    try:
        float(str(num).replace(",", ""))
        return True
    except:
        return False


def numeric_equal(prediction, reference, tol=1e-4):
    """Check if two numbers are approximately equal."""
    try:
        return abs(float(prediction) - float(reference)) < tol
    except:
        return False


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """
    Mathematical equality check with support for various formats.
    Based on grader.py implementation.
    """
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    try:  # Numerical equality
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)

            if pred_num is None or ref_num is None:
                return False

            # Handle percentage variations
            if include_percentage:
                gt_result = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_result = [ref_num]

            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(pred_num, item):
                            return True
                    else:
                        if item == pred_num:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # String equality after normalization
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Remove brackets and parentheses for comparison
    pred_str, ref_str = prediction, reference
    for s in ["{", "}", "(", ")", "[", "]"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")

    if pred_str.lower() == ref_str.lower():
        return True

    # Try symbolic comparison (simplified)
    try:
        # Remove common mathematical notation differences
        pred_clean = prediction.replace(" ", "").lower()
        ref_clean = reference.replace(" ", "").lower()
        if pred_clean == ref_clean:
            return True
    except:
        pass

    return False


def extract_boxed_answer(text: str) -> str:
    """Extract answer from LaTeX \\boxed{} notation."""
    if "\\boxed{" in text and "}" in text:
        answer = text.split("\\boxed{")[-1]
        last_index = answer.rfind("}")
        return answer[:last_index].strip()
    else:
        return text.strip()


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags or \\boxed{} notation."""
    if "<answer>" in text and "</answer>" in text:
        after_ans = text.split("<answer>")[-1]
        answer = after_ans.split("</answer>")[0]
        unboxed_answer = extract_boxed_answer(answer)
        return unboxed_answer.strip()
    else:
        # Sometimes it is in "\\boxed{answer}"
        return extract_boxed_answer(text)


def has_answer_tags(text: str) -> bool:
    """Check if text has proper answer tags."""
    return "<answer>" in text and "</answer>" in text


def has_all_format(text: str) -> bool:
    """Check if text follows the complete expected format."""
    # Criteria 1: <think> should be present
    criteria_1 = "<think>" in text and "</think>" in text
    # There should only be one </think>
    criteria_2 = text.count("</think>") == 1
    # Think should be the first thing in the text
    text_stripped = text.strip()
    criteria_3 = text_stripped.startswith("<think>")
    # </think> should come before <answer>
    criteria_4 = "</think>" in text and text.index("</think>") < text.index("<answer>") if "<answer>" in text else True
    # After </think> and before <answer> stripped should be empty
    criteria_5 = (
        text.strip().split("</think>")[1].split("<answer>")[0].strip() == ""
        if criteria_1 and criteria_2 and criteria_3 and criteria_4
        else False
    )
    return criteria_1 and criteria_2 and criteria_3 and criteria_4 and criteria_5


def compute_soft_format_reward(text: str) -> float:
    """Compute soft format reward based on partial compliance."""
    reward = 0.0
    MAX_SOFT_REWARD = 1.0
    NUMBER_OF_SOFT_REWARDS = 8
    value_increment = MAX_SOFT_REWARD / NUMBER_OF_SOFT_REWARDS

    # If starts with <
    if text.strip().startswith("<"):
        reward += value_increment
    # If first token is <think>
    if text.strip().startswith("<think>"):
        reward += value_increment
    # If there is exactly one <think>
    if text.count("<think>") == 1:
        reward += value_increment
    # If there is exactly one </think>
    if text.count("</think>") == 1:
        reward += value_increment
    # If there is exactly one <answer>
    if text.count("<answer>") == 1:
        reward += value_increment
    # If there is exactly one </answer>
    if text.count("</answer>") == 1:
        reward += value_increment
    # We didn't ask for boxed{}, reward if it's not there
    if "\\boxed{" not in text:
        reward += value_increment
    # No text between </think> and <answer>
    if "</think>" in text and "<answer>" in text:
        between_tags = text.split("</think>")[1].split("<answer>")[0]
        if between_tags.strip() == "":
            reward += value_increment

    return reward


def compute_length_penalty(is_correct: bool, all_lengths: list[int], current_length: int) -> float:
    """
    Compute length penalty based on relative length within batch.

    Given k sampled responses, computes:
    len_reward(i) = λ if correct, min(0, λ) if incorrect
    where λ = 0.5 - (len(i) - min_len) / (max_len - min_len)

    This promotes shorter responses and penalizes longer responses among correct ones,
    while explicitly penalizing long responses with incorrect answers.
    """

    # Batch-aware length penalty implementation
    min_len = min(all_lengths)
    max_len = max(all_lengths)

    # If all responses have same length, set length reward to zero
    if max_len == min_len:
        return 0.0

    # Compute λ = 0.5 - (len(i) - min_len) / (max_len - min_len)
    lambda_val = 0.5 - ((current_length - min_len) / (max_len - min_len))

    if is_correct:
        # For correct answers, give λ reward (can be positive or negative)
        return lambda_val
    else:
        # if wrong, just give 0
        return 0.0
        # For incorrect answers, give min(0, λ) reward (only penalty, no reward)
        # return min(0, lambda_val)


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict[str, str | float | bool]:
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

    # Ensure inputs are strings
    if not isinstance(solution_str, str):
        solution_str = str(solution_str)
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    # 1. Basic answer format reward (0.5 points)
    # Reward if <answer> and </answer> tags are present
    if has_answer_tags(solution_str):
        total_reward += 0.5

    # 2. Full format compliance reward (0.5 points)
    # Reward if follows complete format with <think> tags
    if has_all_format(solution_str):
        total_reward += 0.5

    # 3. Soft format rewards (1.0 points)
    # Partial credit for format compliance
    soft_reward = compute_soft_format_reward(solution_str)
    total_reward += soft_reward

    # 4. Correctness reward (1.0 points)
    extracted_answer = extract_answer(solution_str)
    is_correct = math_equal(extracted_answer, ground_truth)

    # Extract batch length information from extra_info
    assert extra_info is not None and isinstance(extra_info, dict), "extra_info must be a non-None dictionary"
    assert "all_lengths" in extra_info, "extra_info must contain 'all_lengths' key"
    all_lengths: list[int] = extra_info["all_lengths"]
    assert "response_length" in extra_info, "extra_info must contain 'response_length' key"
    response_length: int = extra_info["response_length"]

    if is_correct:
        total_reward += 1.0
        # 5. Length penalty (up to 0.25 points)
        length_reward = 0.5 * compute_length_penalty(
            is_correct=True, all_lengths=all_lengths, current_length=response_length
        )
        total_reward += length_reward
    else:
        length_reward = 0.5 * compute_length_penalty(
            is_correct=False, all_lengths=all_lengths, current_length=response_length
        )
        total_reward += length_reward

    return {"score": total_reward, "is_correct": is_correct}
