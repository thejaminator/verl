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
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py

from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/grader.py
"""

import re
from math import isclose

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def choice_answer_clean(pred: str) -> str:
    cleaned_pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp: list[str] = re.findall(r"\b(A|B|C|D|E)\b", cleaned_pred.upper())
    if tmp:
        answer_list = tmp
    else:
        answer_list = [cleaned_pred.strip().strip(".")]
    final_answer = answer_list[-1]
    # Remove the period at the end, again!
    final_answer = final_answer.rstrip(".").rstrip("/")
    return final_answer


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:  # noqa: E722
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:  # noqa: E722
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:  # noqa: E722
                try:
                    return f(s)
                except:  # noqa: E722
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:  # noqa: E722
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:  # noqa: E722
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:  # noqa: E722
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:  # noqa: E722
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:  # noqa: E722
        pass

    return False


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:  # noqa: E722
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]
            ):
                return True
    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines, strict=False):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    # symbolic equal with sympy
    if symbolic_equal(prediction, reference):
        return True

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

    return {
        "score": total_reward,
        "is_correct": is_correct,
        "parsed_answer": extracted_answer,
    }


if __name__ == "__main__":
    result = math_equal("-1/9", "-\\frac{1}{9}")
    print(result)
