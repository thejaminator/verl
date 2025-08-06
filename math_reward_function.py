# ruff: noqa: E722
"""
From will brown's
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


# Dan Hendrycks' code
def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    # replace power of subscript ²,³,⁴,⁵,⁶,⁷,⁸,⁹,⁰
    expr = expr.replace("⁰", "^0")
    expr = expr.replace("¹", "^1")
    expr = expr.replace("²", "^2")
    expr = expr.replace("³", "^3")
    expr = expr.replace("⁴", "^4")
    expr = expr.replace("⁵", "^5")
    expr = expr.replace("⁶", "^6")
    expr = expr.replace("⁷", "^7")
    expr = expr.replace("⁸", "^8")
    expr = expr.replace("⁹", "^9")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def grade_answer(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth:
        return False
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = solution_str
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)


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

    # Compute λ = 1 - (len(i) - min_len) / (max_len - min_len)
    lambda_val = 1 - ((current_length - min_len) / (max_len - min_len))

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
    is_correct = grade_answer(extracted_answer, ground_truth)

    # Extract batch length information from extra_info
    assert extra_info is not None and isinstance(extra_info, dict), "extra_info must be a non-None dictionary"
    assert "all_lengths" in extra_info, "extra_info must contain 'all_lengths' key"
    all_lengths: list[int] = extra_info["all_lengths"]
    assert "response_length" in extra_info, "extra_info must contain 'response_length' key"
    response_length: int = extra_info["response_length"]

    if is_correct:
        total_reward += 1.0
        # 5. Length penalty (up to 0.25 points)
        length_reward = 0.25 * compute_length_penalty(
            is_correct=True, all_lengths=all_lengths, current_length=response_length
        )
        total_reward += length_reward
    else:
        length_reward = 0.25 * compute_length_penalty(
            is_correct=False, all_lengths=all_lengths, current_length=response_length
        )
        total_reward += length_reward

    return {
        "score": total_reward,
        "is_correct": is_correct,
        "parsed_answer": extracted_answer,
    }


if __name__ == "__main__":
    result = grade_answer("-1/9", "-\\frac{1}{9}")
    print(result)
    result = grade_answer("y = 3x² - 34x + 88", "y = 3x^2 - 34x + 88")
    print(result)
    expr = grade_answer("21,000", "21,\!000")
    print(expr)
    result = grade_answer("3/5", "\\frac{3}{5}")
    print(result)
