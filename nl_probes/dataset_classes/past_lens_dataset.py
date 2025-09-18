import gc
import os
import pickle
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Generator, Literal

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from detection_eval.steering_hooks import get_introspection_prefix
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)


@dataclass
class PastLensDatasetConfig(BaseDatasetConfig):
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    min_k_activations: int = 1
    max_k_activations: int = 20
    batch_size: int = 128
    max_length: int = 512


class PastLensDatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"

        self.dataset_config.dataset_name = "past_lens"

        self.dataset_params: PastLensDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Past lens dataset only supports train split right now"
        assert self.dataset_config.num_test == 0, "Past lens dataset only supports train split right now"

        if self.dataset_config.num_train < self.dataset_params.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be greater than or equal to batch_size {self.dataset_params.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        dataset = hf_mixed_dataset_to_generator(tokenizer)

        dtype = torch.bfloat16

        training_data = collect_past_lens_acts(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=tokenizer,
            dataset=dataset,
            num_datapoints=self.dataset_config.num_train,
            dtype=dtype,
        )

        self.save_dataset(training_data, "train")


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    min_chars: int = 1,
    pretrain_frac: float = 0.9,  # 0.9 → 90 % pretrain, 10 % chat
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    chat_key: str = "conversation",
    sequence_pack_pretrain: bool = False,
    sequence_pack_chat: bool = False,
):
    """Get a mix of pretrain and chat data at a specified ratio. By default, 90% of the data will be pretrain and 10% will be chat.

    Default datasets:
    pretrain_dataset: "HuggingFaceFW/fineweb"
    chat_dataset: "lmsys/lmsys-chat-1m"

    Note that you will have to request permission for lmsys (instant approval on HuggingFace).

    min_chars: minimum number of characters per sample. To perform sequence packing, set it to ~4x sequence length in tokens.
    Samples will be joined with the eos token.
    If it's low (like 1), each sample will just be a single row from the dataset, padded to the max length. Sometimes this will fill the context, sometimes it won't.

    Why use strings instead of tokens? Because dictionary learning expects an iterator of strings, and this is simple and good enough.

    Implicit assumption: each sample will be truncated to sequence length when tokenized.

    By default, we sequence pack the pretrain data and DO NOT sequence pack the chat data, as it would look kind of weird. The EOS token is used to separate
    user / assistant messages, not to separate conversations from different users.
    If you want to sequence pack the chat data, set sequence_pack_chat to True.

    Pretrain format will be: <bos>text<eos>text<eos>text<eos>...
    Chat format will be <formatted chat message> Optionally: <formatted chat message><formatted chat message>...

    Other parameters:
    - system_prompt_to_remove: an optional string that will be removed from the chat data with a given frequency.
        You probably want to verify that the system prompt you pass in is correct.
    - system_prompt_removal_freq: the frequency with which the system prompt will be removed

    Why? Well, we probably don't want to have 1000's of copies of the system prompt in the training dataset. But we also may not want to remove it entirely.
    And we may want to use the LLM with no system prompt when comparing between models.
    IDK, this is a complicated and annoying detail. At least this constrains the complexity to the dataset generator.
    """
    if not 0 < pretrain_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    assert min_chars > 0

    # Load both datasets as iterable streams
    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    # Convert the fraction to two small integers (e.g. 0.9 → 9 / 10)
    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain
    eos_token = tokenizer.eos_token

    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token

    def gen():
        while True:
            for _ in range(n_pretrain):
                if sequence_pack_pretrain:
                    length = 0
                    samples = []
                    while length < min_chars:
                        # Add bos token to the beginning of the sample
                        sample = next(pretrain_ds)[pretrain_key]
                        samples.append(sample)
                        length += len(sample)
                    samples = bos_token + eos_token.join(samples)
                    yield samples
                else:
                    sample = bos_token + next(pretrain_ds)[pretrain_key]
                    yield sample
            for _ in range(n_chat):
                if sequence_pack_chat:
                    length = 0
                    samples = []
                    while length < min_chars:
                        sample = next(chat_ds)[chat_key]
                        # Apply chat template also includes bos token
                        sample = tokenizer.apply_chat_template(sample, tokenize=False)
                        samples.append(sample)
                        length += len(sample)
                    samples = "".join(samples)
                    yield samples
                else:
                    sample = tokenizer.apply_chat_template(next(chat_ds)[chat_key], tokenize=False)
                    yield sample

    return gen()


def collect_past_lens_acts(
    dataset_config: DatasetLoaderConfig,
    custom_dataset_params: PastLensDatasetConfig,
    tokenizer: AutoTokenizer,
    dataset: Generator,
    num_datapoints: int,
    dtype: torch.dtype,
) -> list[TrainingDataPoint]:
    random.seed(dataset_config.seed)
    torch.manual_seed(dataset_config.seed)

    layers = [
        layer_percent_to_layer(dataset_config.model_name, layer_percent)
        for layer_percent in dataset_config.layer_percents
    ]

    device = torch.device("cpu")
    if dataset_config.save_acts:
        model = load_model(dataset_config.model_name, dtype)
        submodules = {layer: get_hf_submodule(model, layer) for layer in layers}
        device = model.device

    training_data = []

    for i in tqdm(range(0, num_datapoints, custom_dataset_params.batch_size), desc="Collecting past lens acts"):
        inputs = []
        for _ in range(custom_dataset_params.batch_size):
            inputs.append(next(dataset))

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=custom_dataset_params.max_length,
            add_special_tokens=False,
        ).to(device)

        if dataset_config.save_acts:
            acts_BLD_by_layer_dict = collect_activations_multiple_layers(
                model, submodules, tokenized_inputs, min_offset=None, max_offset=None
            )
            for layer in acts_BLD_by_layer_dict.keys():
                acts_BLD_by_layer_dict[layer] = acts_BLD_by_layer_dict[layer].to("cpu", non_blocking=True)

        attn_mask_BL = tokenized_inputs["attention_mask"].cpu()
        input_ids_BL = tokenized_inputs["input_ids"].cpu()

        for layer in layers:
            for j in range(len(inputs)):
                attn_mask_L = attn_mask_BL[j].bool()
                input_ids_L = input_ids_BL[j, attn_mask_L]
                L = len(input_ids_L)

                k_prev = random.randint(custom_dataset_params.min_k_tokens, custom_dataset_params.max_k_tokens)
                k_acts = random.randint(
                    custom_dataset_params.min_k_activations, custom_dataset_params.max_k_activations
                )

                # Skip if sequence is too short (less than k+1 valid tokens)
                if len(input_ids_L) < k_prev + k_acts + 1:
                    continue

                # Select a random position that's at least k tokens from the start
                selected_act_begin_idx = random.randint(k_prev, L - k_acts - 1)
                selected_act_positions = list(range(selected_act_begin_idx, selected_act_begin_idx + k_acts))
                selected_tokens_positions = list(range(selected_act_begin_idx - k_prev, selected_act_begin_idx))
                max_position = selected_act_positions[-1]
                input_ids_L = input_ids_L[: max_position + 1]

                past_tokens = input_ids_L[selected_tokens_positions]
                past_text = tokenizer.decode(past_tokens, skip_special_tokens=True)
                prompt = f"Can you predict the previous {k_prev} tokens that came before this?"

                if dataset_config.save_acts:
                    acts_LD = acts_BLD_by_layer_dict[layer][j, attn_mask_L]
                    selected_acts_KD = acts_LD[selected_act_positions]
                    assert selected_acts_KD.shape[0] == k_acts
                    context_input_ids = None
                    context_positions = None
                else:
                    selected_acts_KD = None
                    context_input_ids = input_ids_L
                    context_positions = selected_act_positions

                training_data_point = create_training_datapoint(
                    datapoint_type=dataset_config.dataset_name,
                    prompt=prompt,
                    target_response=past_text,
                    layer=layer,
                    num_positions=k_acts,
                    tokenizer=tokenizer,
                    acts_BD=selected_acts_KD,
                    feature_idx=-1,
                    context_input_ids=context_input_ids,
                    context_positions=context_positions,
                )
                training_data.append(training_data_point)

    return training_data


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # print("Loading dataset...")
    # raise Exception("Stop here")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    layer_percents = [25, 50, 75]

    batch_size = 128
    num_datapoints = 600_000
    max_length = 512
    min_k = 3
    max_k = 20
    seed = 42
    sft_data_folder = "sft_training_data"

    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=PastLensDatasetConfig(),
        dataset_folder=sft_data_folder,
        num_train=num_datapoints,
        num_test=0,
        splits=["train"],
        model_name=model_name,
        layer_percents=layer_percents,
        seed=seed,
        save_acts=True,
    )

    past_lens_dataset_loader = PastLensDatasetLoader(
        dataset_config=dataset_config,
    )
    past_lens_dataset_loader.create_dataset()
