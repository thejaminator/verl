import asyncio
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import detection_eval.caller as caller
from detection_eval.detection_basemodels import SAEInfo
from detection_eval.steering_hooks import EXPLANATION_PROMPT
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.sae import BaseSAE, get_sae_info, load_max_acts_data, load_sae
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.dataset_utils import (
    SAEExplained,
    TrainingDataPoint,
    TrainingExample,
    create_training_datapoint,
    load_explanations_from_jsonl,
)

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


@dataclass
class SAEActivatingSequencesDatasetConfig(BaseDatasetConfig):
    sae_repo_id: str
    use_decoder_vectors: bool = True
    num_features: int = 60_000
    max_examples_per_feature: int = 5
    context_length: int = 32
    verbose: bool = False


@dataclass
class SAEYesNoDatasetConfig(BaseDatasetConfig):
    sft_data_file: str
    llm_model_name: str = "gpt-5-mini-2025-08-07"
    temperature: float = 1.0
    max_tokens: int = 2_000
    max_features: int | None = None
    use_decoder_vectors: bool = True
    max_parallel_requests: int = 100
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    verbose: bool = False


@dataclass
class SAEExplanationDatasetConfig(BaseDatasetConfig):
    sft_data_file: str
    use_decoder_vectors: bool = True
    positive_negative_examples: bool = False
    input_prompt: str = EXPLANATION_PROMPT


YES_NO_QUESTION_PROMPT = """
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


class SAEActivatingSequencesDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        if not isinstance(dataset_config.custom_dataset_params, SAEActivatingSequencesDatasetConfig):
            raise TypeError("Expected SAEActivatingSequencesDatasetConfig")
        self.dataset_params: SAEActivatingSequencesDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden for SAE activating sequences"

        dataset_name = f"sae_activating_sequences_{self.dataset_params.sae_repo_id}_layer_percent_{self.dataset_config.layer_percents[0]}"
        self.dataset_config.dataset_name = dataset_name

        assert self.dataset_config.splits == ["train"], "SAE activating sequences dataset only supports train split"
        assert self.dataset_config.num_test == 0, "SAE activating sequences dataset does not support a test split"
        assert len(self.dataset_config.layer_percents) == 1, (
            "SAE activating sequences dataset only supports one layer percent"
        )

    def create_dataset(self) -> None:
        training_data, sae_info = create_activating_sequences_data(
            datapoint_type=self.dataset_config.dataset_name,
            model_name=self.dataset_config.model_name,
            sae_repo_id=self.dataset_params.sae_repo_id,
            sae_layer_percent=self.dataset_config.layer_percents[0],
            use_decoder=self.dataset_params.use_decoder_vectors,
            num_features=self.dataset_params.num_features,
            max_num_examples=self.dataset_params.max_examples_per_feature,
            seed=self.dataset_config.seed,
            sft_data_folder=self.dataset_config.dataset_folder,
            verbose=self.dataset_params.verbose,
        )

        self.save_dataset(training_data, "train")


class SAEYesNoDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        if not isinstance(dataset_config.custom_dataset_params, SAEYesNoDatasetConfig):
            raise TypeError("Expected SAEYesNoDatasetConfig")

        assert self.dataset_config.splits == ["train"], "SAE explanation dataset only supports train split"
        assert self.dataset_config.num_test == 0, "SAE explanation dataset does not support a test split"
        assert len(self.dataset_config.layer_percents) == 1, "SAE explanation dataset only supports one layer percent"

        self.dataset_params: SAEYesNoDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden for SAE Yes/No dataset"

        self.dataset_config.dataset_name = f"sae_yes_no_{self.dataset_params.sft_data_file}"

    def create_dataset(self) -> None:
        training_data, sae_info = create_yes_no_data(
            model_name=self.dataset_config.model_name,
            dataset_type=self.dataset_config.dataset_name,
            sft_data_file=self.dataset_params.sft_data_file,
            sft_data_folder=self.dataset_config.dataset_folder,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            seed=self.dataset_config.seed,
            use_decoder=self.dataset_params.use_decoder_vectors,
            max_features=self.dataset_params.max_features,
            verbose=self.dataset_params.verbose,
        )

        self.save_dataset(training_data, "train")


class SAEExplanationDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        if not isinstance(dataset_config.custom_dataset_params, SAEExplanationDatasetConfig):
            raise TypeError("Expected SAEExplanationDatasetConfig")
        assert self.dataset_config.splits == ["train"], "SAE explanation dataset only supports train split"
        assert self.dataset_config.num_test == 0, "SAE explanation dataset does not support a test split"
        assert len(self.dataset_config.layer_percents) == 1, "SAE explanation dataset only supports one layer percent"

        self.dataset_params: SAEExplanationDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden for SAE explanation dataset"

        self.dataset_config.dataset_name = f"sae_explanations_{self.dataset_params.sft_data_file}"

    def create_dataset(self) -> None:
        training_data, sae_info = load_sae_data_from_sft_data_file(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=load_tokenizer(self.dataset_config.model_name),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        self.save_dataset(training_data, "train")


def create_activating_sequences_data(
    datapoint_type: str,
    model_name: str,
    sae_repo_id: str,
    sae_layer_percent: int,
    use_decoder: bool = True,
    num_features: int = 60_000,
    max_num_examples: int = 5,
    seed: int = 42,
    sft_data_folder: str = "sft_training_data",
    verbose: bool = False,
) -> tuple[list[TrainingDataPoint], SAEInfo]:
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

    for feature_idx in tqdm(range(num_features), desc="Creating training data"):
        if use_decoder:
            vector = sae.W_dec[feature_idx].clone()
        else:
            vector = sae.W_enc[:, feature_idx].clone()

        vector_1D = vector.unsqueeze(0)

        num_examples = random.randint(2, max_num_examples)

        prompt = TEMPLATES[random.randint(0, len(TEMPLATES) - 1)].format(k=num_examples)

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
                datapoint_type=datapoint_type,
                prompt=prompt,
                target_response=output,
                layer=sae_info.sae_layer,
                num_positions=1,
                tokenizer=tokenizer,
                acts_BD=vector_1D,
                feature_idx=feature_idx,
            )
        )

    return training_data, sae_info


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
    dataset_type: str,
    sft_data_file: str,
    sft_data_folder: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
    use_decoder: bool = True,
    max_features: int | None = None,
    verbose: bool = False,
) -> tuple[list[TrainingDataPoint], SAEInfo]:
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

    tokenizer = load_tokenizer(model_name)

    incorrect_count = 0

    for explanation, response in zip(explanations, responses):
        feature_idx = explanation.sae_id

        if use_decoder:
            vector = sae.W_dec[feature_idx].clone()
        else:
            vector = sae.W_enc[:, feature_idx].clone()

        vector_1D = vector.unsqueeze(0)

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
            question_prompt = f"Answer with 'Yes' or 'No' only. {qa['question']}"
            training_datapoint = create_training_datapoint(
                datapoint_type=dataset_type,
                prompt=question_prompt,
                target_response=qa["answer"],
                layer=sae_info.sae_layer,
                num_positions=1,
                tokenizer=tokenizer,
                acts_BD=vector_1D,
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
    print(f"Incorrect count: {incorrect_count}")

    return training_data, sae_info


@torch.no_grad()
def construct_train_dataset(
    custom_dataset_params: SAEExplanationDatasetConfig,
    dataset_type: str,
    dataset_size: int,
    layer: int,
    input_prompt: str,
    training_examples: list[TrainingExample],
    sae: BaseSAE,
    tokenizer: AutoTokenizer,
) -> list[TrainingDataPoint]:
    training_data = []

    for i in tqdm(range(dataset_size), desc="Constructing training dataset"):
        target_response = training_examples[i].explanation
        target_feature_idx = training_examples[i].feature_idx
        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        # .clone() because otherwise we will save the entire W_dec in pickle for each training example
        if custom_dataset_params.use_decoder_vectors:
            feature_vector = sae.W_dec[target_feature_idx].clone()
        else:
            feature_vector = sae.W_enc[:, target_feature_idx].clone()

        feature_vector_1D = feature_vector.unsqueeze(0)

        training_data_point = create_training_datapoint(
            datapoint_type=dataset_type,
            prompt=input_prompt,
            target_response=target_response,
            layer=layer,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=feature_vector_1D,
            feature_idx=target_feature_idx,
        )

        if i == 0:
            # Fully print the first example
            print("First training example:")
            print(f"prompt: {input_prompt}")
            print(f"target_response: {target_response}")
            print("-" * 100)

        training_data.append(training_data_point)

    return training_data


def construct_eval_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    eval_feature_indices: list[int],
    api_data: dict,
    sae: BaseSAE,
    tokenizer: AutoTokenizer,
    enable_thinking: bool = False,
) -> list[TrainingDataPoint]:
    """Every prompt is exactly the same - the only difference is the steering vectors."""

    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=enable_thinking,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")
    # Set [-100] for user prompts
    labels = [-100] * len(input_prompt_ids)

    orig_prompt_length = len(input_prompt_ids)

    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    eval_data = []

    first_position = None

    for i in tqdm(range(dataset_size), desc="Constructing eval dataset"):
        target_feature_idx = eval_feature_indices[i]

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        if cfg.use_decoder_vectors:
            feature_vector = sae.W_dec[target_feature_idx].clone()
        else:
            feature_vector = sae.W_enc[:, target_feature_idx].clone()

        positions = []
        for i in range(orig_prompt_length):
            if input_prompt_ids[i] == x_token_id:
                positions.append(i)

        assert len(positions) == 1, "Expected exactly one X token"

        if first_position is None:
            first_position = positions[0]
        else:
            assert positions[0] == first_position, "Expected all positions to be the same"
        assert len(input_prompt_ids) > 0

        eval_data_point = TrainingDataPoint(
            datapoint_type="eval",
            input_ids=input_prompt_ids,
            labels=labels,
            layer=sae.hook_layer,
            steering_vectors=feature_vector,
            positions=positions,
            feature_idx=target_feature_idx,
            target_output="",
            context_input_ids=None,
            context_positions=None,
        )

        eval_data.append(eval_data_point)

    return eval_data


def load_sae_data_from_sft_data_file(
    dataset_config: DatasetLoaderConfig,
    custom_dataset_params: SAEExplanationDatasetConfig,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[TrainingDataPoint], SAEInfo]:
    explanations: list[SAEExplained] = load_explanations_from_jsonl(custom_dataset_params.sft_data_file)
    orig_sae_info = explanations[0].sae_info
    for data_point in explanations:
        assert data_point.sae_info == orig_sae_info
    sae_info = SAEInfo.model_validate(orig_sae_info)

    sae = load_sae(
        sae_info.sae_repo_id, sae_info.sae_filename, sae_info.sae_layer, dataset_config.model_name, device, dtype
    )

    training_examples = [
        TrainingExample.with_positive_and_negative_examples(exp)
        if custom_dataset_params.positive_negative_examples
        else TrainingExample.with_explanation_only(exp)
        for exp in explanations
    ]
    print(f"Loaded {len(training_examples)} training examples from {custom_dataset_params.sft_data_file}")

    train_features = set()

    for example in training_examples:
        train_features.add(example.feature_idx)

    # For evaluation, we'll use a subset of the training features
    # In a real scenario, you might want to load a separate eval set
    print(f"train examples: {len(training_examples)}")
    print(f"Train features: {len(train_features)}")

    training_data: list[TrainingDataPoint] = construct_train_dataset(
        custom_dataset_params,
        dataset_type=dataset_config.dataset_name,
        dataset_size=len(training_examples),
        # dataset_size,
        layer=sae_info.sae_layer,
        input_prompt=EXPLANATION_PROMPT,
        training_examples=training_examples,
        sae=sae,
        tokenizer=tokenizer,
    )

    return training_data, sae_info


# if __name__ == "__main__":
#     # cfg.sae_repo_id = "fnlp/Llama3_1-8B-Base-LXR-32x"
#     # cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"

#     device = torch.device("cpu")
#     dtype = torch.bfloat16

#     sae_repo_id = "adamkarvonen/qwen3-8b-saes"
#     model_name = "Qwen/Qwen3-8B"
#     sft_data_folder = "sft_training_data"
#     os.makedirs(sft_data_folder, exist_ok=True)

#     sae_layer_percents = [25, 50, 75]
#     # sae_layer_percents = [75]
#     # sae_layer_percents = [25]
#     num_features = 60_000

#     for layer_percent in sae_layer_percents:
#         create_activating_sequences_data(model_name, sae_repo_id, layer_percent, num_features=num_features)

#     for layer_percent in sae_layer_percents:
#         explanations_file = (
#             f"data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
#         )

#         create_yes_no_data(
#             model_name,
#             explanations_file,
#             sft_data_folder,
#             device,
#             dtype,
#             seed=42,
#             use_decoder=True,
#             max_features=None,
#             # max_features=10,
#             verbose=False,
#             # verbose=True,
#         )
