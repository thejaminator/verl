import gc
import os
import pickle
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Generator, Literal

import torch
from dataset_classes.act_dataset_manager import ActDatasetLoader
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
    min_k: int = 3
    max_k: int = 20
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

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        dataset = hf_mixed_dataset_to_generator(tokenizer)

        device = torch.device("cuda")
        dtype = torch.bfloat16

        save_path = os.path.join(self.dataset_config.dataset_folder, self.get_dataset_filename("train"))

        training_data = collect_past_lens_acts(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=tokenizer,
            dataset=dataset,
            num_datapoints=self.dataset_config.num_train,
            device=device,
            dtype=dtype,
        )

        torch.save(
            {
                "config": asdict(self.dataset_config),
                "data": [dp.model_dump() for dp in training_data],
            },
            save_path,
        )
        print(f"Saved {len(training_data)} datapoints to {save_path}")


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
    device: torch.device,
    dtype: torch.dtype,
) -> list[TrainingDataPoint]:
    random.seed(dataset_config.seed)
    torch.manual_seed(dataset_config.seed)

    layers = [
        layer_percent_to_layer(dataset_config.model_name, layer_percent)
        for layer_percent in dataset_config.layer_percents
    ]

    model = load_model(dataset_config.model_name, dtype)

    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}

    training_data = []

    for i in tqdm(range(0, num_datapoints, custom_dataset_params.batch_size), desc="Collecting past lens acts"):
        inputs = []
        for _ in range(custom_dataset_params.batch_size):
            inputs.append(next(dataset))

        tokenized_inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=custom_dataset_params.max_length
        ).to(device)

        acts_BLD_by_layer_dict = collect_activations_multiple_layers(
            model, submodules, tokenized_inputs, min_offset=None, max_offset=None
        )

        attn_mask_BL = tokenized_inputs["attention_mask"].cpu()
        input_ids_BL = tokenized_inputs["input_ids"].cpu()

        for layer in layers:
            acts_BLD_by_layer_dict[layer] = acts_BLD_by_layer_dict[layer].cpu()
            acts_BLD = acts_BLD_by_layer_dict[layer]
            for j in range(len(inputs)):
                attn_mask_L = attn_mask_BL[j]
                input_ids_L = input_ids_BL[j]
                acts_LD = acts_BLD[j]

                k = random.randint(custom_dataset_params.min_k, custom_dataset_params.max_k)

                # Find the first non-padding position (where attention mask is 1)
                valid_positions = torch.where(attn_mask_L == 1)[0]

                # Skip if sequence is too short (less than k+1 valid tokens)
                if len(valid_positions) < k + 1:
                    continue

                # Select a random position that's at least k tokens from the start
                # valid_positions[k:] gives us positions starting from the (k+1)th valid token
                valid_indices = valid_positions[k:]  # Positions at least k tokens from start

                # Randomly select one of these valid positions
                selected_idx = valid_indices[random.randint(0, len(valid_indices) - 1)].item()

                # Get the activation at the selected position
                selected_act = acts_LD[selected_idx]  # Shape: [D]
                selected_act_1D = selected_act.unsqueeze(0)  # (1, D)

                past_tokens = input_ids_L[selected_idx - k : selected_idx]

                past_text = tokenizer.decode(past_tokens)

                prompt = f"Can you predict the previous {k} tokens that came before this?"

                training_data_point = create_training_datapoint(
                    prompt=prompt,
                    target_response=past_text,
                    layer=layer,
                    num_positions=1,
                    tokenizer=tokenizer,
                    acts_BD=selected_act_1D,
                    feature_idx=-1,
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
