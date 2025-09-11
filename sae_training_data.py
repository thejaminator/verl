import asyncio
import os
import pickle
import random
import re

import torch
from slist import Slist
from tqdm import tqdm
from transformers import AutoTokenizer

import detection_eval.caller as caller
from classification import create_training_datapoint
from create_hard_negatives_v2 import get_sae_info, load_max_acts_data, load_sae, load_tokenizer
from detection_eval.detection_basemodels import SAEInfo
from detection_eval.steering_hooks import get_introspection_prefix
from sft_config import SAEExplained, TrainingDataPoint, create_training_datapoint, load_explanations_from_jsonl

TEMPLATES: list[str] = [
    "Give {k} short sentences that would strongly activate the concept.",
    "Write {k} realistic snippets that you expect to trigger this feature strongly.",
    "List {k} lines that should cause high activation for the concept.",
    "Produce {k} example sentences likely to fire this feature.",
    "Return {k} candidate sentences that maximally activate it.",
    "Generate {k} examples that strongly express the concept.",
    "Provide {k} natural sentences that would make this feature turn on.",
    "Output {k} minimal sentences that strongly activate the underlying idea.",
    "Write {k} short texts that highly activate the concept in context.",
    "Give {k} examples that should peak this feature's activation.",
    "List {k} phrases that would robustly trigger the concept.",
    "Produce {k} simple sentences expected to yield strong activation.",
]


def create_activating_sequences_data(
    model_name: str,
    sae_repo_id: str,
    sae_layer_percent: int,
    use_decoder: bool = True,
    num_features: int = 60_000,
    max_num_examples: int = 5,
    seed: int = 42,
    sft_data_folder: str = "sft_training_data",
    verbose: bool = False,
):
    device = torch.device("cpu")
    dtype = torch.bfloat16
    random.seed(seed)

    tokenizer = load_tokenizer(model_name)

    training_data = []

    sae_info = get_sae_info(sae_repo_id, sae_layer_percent)

    save_filename = f"act_examples_{model_name}_layer_percent_{sae_info.sae_layer_percent}_width_{sae_info.sae_width}_num_features_{num_features}"
    save_filename = save_filename.replace("/", "_").replace(".", "_").replace(" ", "_")
    save_filename = f"{sft_data_folder}/{save_filename}.pkl"

    sae = load_sae(sae_info.sae_repo_id, sae_info.sae_filename, sae_info.sae_layer, model_name, device, dtype)

    max_acts_data = load_max_acts_data(
        model_name=model_name,
        sae_layer=sae_info.sae_layer,
        sae_width=sae_info.sae_width,
        layer_percent=sae_info.sae_layer_percent,
        context_length=32,
    )

    prefix = get_introspection_prefix(sae_info.sae_layer)

    for feature_idx in tqdm(range(num_features), desc="Creating training data"):
        if use_decoder:
            vector = sae.W_dec[feature_idx].clone()
        else:
            vector = sae.W_enc[:, feature_idx].clone()

        num_examples = random.randint(2, max_num_examples)

        prompt = TEMPLATES[random.randint(0, len(TEMPLATES) - 1)].format(k=num_examples)
        prompt = f"{prefix}{prompt}"

        output = ""

        tokens_BL = max_acts_data["max_tokens"][feature_idx, :num_examples]
        acts_BL = max_acts_data["max_acts"][feature_idx, :num_examples]

        max_acts_B = acts_BL.max(dim=-1).values
        if max_acts_B[-1] <= 0:
            continue

        for i in range(num_examples):
            text = tokenizer.decode(tokens_BL[i], skip_special_tokens=True)
            output += f"Example {i + 1}: {text}\n"

        if verbose:
            print(f"prompt: {prompt}")
            print(f"target_response: {output}")
            print("-" * 100)

        training_data.append(
            create_training_datapoint(
                prompt=prompt,
                target_response=output,
                tokenizer=tokenizer,
                acts_D=vector,
                feature_idx=feature_idx,
            )
        )

    with open(save_filename, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} datapoints to {save_filename}")


def parse_yes_no_qas(response: str) -> list[dict[str, str]] | None:
    """
    Parse an LLM response containing four Q/A pairs in this form:

      <question> ... </question>
      <answer> ... </answer>

    Returns a list of 4 dicts: {"question": str, "answer": "Yes" | "No"}.

    Validations:
      1) exactly 4 Q/A pairs
      2) each answer is Yes or No (case-insensitive, punctuation ignored)
      3) exactly 2 Yes and 2 No

    Returns None on any violation.
    """
    pair_re = re.compile(
        r"<question>\s*(.*?)\s*</question>\s*<answer>\s*(.*?)\s*</answer>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    pairs = pair_re.findall(response)
    if len(pairs) != 4:
        return None

    out: list[dict[str, str]] = []
    yes_count = 0
    no_count = 0

    for q_raw, a_raw in pairs:
        # Normalize whitespace in the question
        question = " ".join(q_raw.strip().split())

        # Normalize the answer to Yes/No
        a_clean = re.sub(r"[^A-Za-z]", "", a_raw).lower()
        if a_clean in {"yes", "y"}:
            answer = "Yes"
            yes_count += 1
        elif a_clean in {"no", "n"}:
            answer = "No"
            no_count += 1
        else:
            return None

        out.append({"question": question, "answer": answer})

    if yes_count != 2 or no_count != 2:
        return None

    return out


def create_yes_no_data(
    model_name: str,
    sft_data_file: str,
    sft_data_folder: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
    use_decoder: bool = True,
    max_features: int | None = None,
    verbose: bool = False,
):
    question_gen_prompt = """
I would like for you to generate four Yes / No questions about the feature's explanation. 2 should have the answer be Yes, and 2 should have the answer be No.

<BEGIN EXAMPLE>

<explanation>
Mentions of named, recurring marquee sports events—especially U.S. college football "Bowl" games and NHL "Classic"/"Winter Classic" events. These sentences typically include event names (e.g., Fiesta Bowl, Orange Bowl, Cotton Bowl Classic, Winter Classic), venue or stadium references, dates or seasonal timing, and verbs/phrases like "played," "host," "has been played annually," or "will be the host." Not general sports commentary or unrelated news—specifically the formal naming/placement/occurrence of annual or special sporting events.
</explanation>

Response:

<question>
Would you say the concept is related to recurring sports events?
</question>
<answer>
Yes
</answer>
<question>
Is this feature most related to general sports commentary?
</question>
<answer>
No
</answer>
<question>
Does this relate to pets, especially dogs?
</question>
<answer>
No
</answer>
<question>
Does this have any relation to sports or college football?
</question>
<answer>
Yes
</answer>

<END EXAMPLE>

Here is the explanation of a sparse autoencoder feature.

<explanation>
{explanation}
</explanation>

Please generate four Yes / No questions, and try have some variety in the phrasing and types of questions.
"""

    random.seed(seed)

    explanations: list[SAEExplained] = load_explanations_from_jsonl(sft_data_file)
    orig_sae_info = explanations[0].sae_info
    for data_point in explanations:
        assert data_point.sae_info == orig_sae_info
    sae_info = SAEInfo.model_validate(orig_sae_info)

    save_filename = f"yes_no_sae_data_{model_name}_layer_percent_{sae_info.sae_layer_percent}_width_{sae_info.sae_width}_max_features_{max_features}"
    save_filename = save_filename.replace("/", "_").replace(".", "_").replace(" ", "_")
    save_filename = f"{sft_data_folder}/{save_filename}.pkl"

    sae = load_sae(sae_info.sae_repo_id, sae_info.sae_filename, sae_info.sae_layer, model_name, device, dtype)

    llm_prompts = []

    if max_features is not None:
        explanations = explanations[:max_features]

    for explanation in explanations:
        prompt = question_gen_prompt.format(explanation=explanation.explanation)
        llm_prompts.append(caller.ChatHistory.from_user(prompt))

    responses = asyncio.run(
        caller.run_list_of_prompts(
            model_name="gpt-5-mini-2025-08-07",
            prompts=llm_prompts,
            temperature=1.0,
            max_tokens=2000,
            max_par=100,
            reasoning_effort="low",
        )
    )

    training_data = []

    prefix = get_introspection_prefix(sae_info.sae_layer)
    tokenizer = load_tokenizer(model_name)

    incorrect_count = 0

    for explanation, response in zip(explanations, responses):
        feature_idx = explanation.sae_id

        if use_decoder:
            vector = sae.W_dec[feature_idx].clone()
        else:
            vector = sae.W_enc[:, feature_idx].clone()

        if verbose:
            print(f"feature_idx: {feature_idx}")
            print(f"explanation: {explanation.explanation}")
            print(response)
            print("-" * 100)

        qas = parse_yes_no_qas(response)
        if qas is None:
            incorrect_count += 1
            continue

        for qa in qas:
            question_prompt = f"{prefix}Answer with 'Yes' or 'No' only. {qa['question']}"
            training_datapoint = create_training_datapoint(
                prompt=question_prompt,
                target_response=qa["answer"],
                tokenizer=tokenizer,
                acts_D=vector,
                feature_idx=feature_idx,
            )

            if training_datapoint is None:
                incorrect_count += 1
                continue

            if verbose:
                print(f"feature_idx: {training_datapoint.feature_idx}")
                print(tokenizer.decode(training_datapoint.input_ids))
                print(training_datapoint.labels)
            training_data.append(training_datapoint)
    with open(save_filename, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} datapoints to {save_filename}")
    print(f"Incorrect count: {incorrect_count}")


if __name__ == "__main__":
    # cfg.sae_repo_id = "fnlp/Llama3_1-8B-Base-LXR-32x"
    # cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"

    device = torch.device("cpu")
    dtype = torch.bfloat16

    sae_repo_id = "adamkarvonen/qwen3-8b-saes"
    model_name = "Qwen/Qwen3-8B"
    sft_data_folder = "sft_training_data"
    os.makedirs(sft_data_folder, exist_ok=True)

    sae_layer_percents = [25, 50, 75]
    # sae_layer_percents = [75]
    # sae_layer_percents = [25]
    num_features = 60_000

    for layer_percent in sae_layer_percents:
        create_activating_sequences_data(model_name, sae_repo_id, layer_percent, num_features=num_features)

    for layer_percent in sae_layer_percents:
        explanations_file = (
            f"data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
        )

        create_yes_no_data(
            model_name,
            explanations_file,
            sft_data_folder,
            device,
            dtype,
            seed=42,
            use_decoder=True,
            max_features=None,
            # max_features=10,
            verbose=False,
            # verbose=True,
        )
