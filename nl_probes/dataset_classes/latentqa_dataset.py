import random
from dataclasses import asdict, dataclass, field
from typing import Generator, Literal

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

import nl_probes.dataset_classes.misc.latentqa_loader as latentqa_loader
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)


@dataclass
class LatentQADatasetConfig(BaseDatasetConfig):
    batch_size: int = 128


class LatentQADatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, Dataset name gets overridden here"
        )

        self.dataset_config.dataset_name = "past_lens"

        self.dataset_params: LatentQADatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Past lens dataset only supports train split right now"
        assert self.dataset_config.num_test == 0, "Past lens dataset only supports train split right now"

        if self.dataset_config.num_train < self.dataset_params.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be greater than or equal to batch_size {self.dataset_params.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        paths = latentqa_loader.DataPaths(
            system=None,
            stimulus_completion="datasets/latentqa_datasets/train/stimulus_completion.json",
            stimulus="datasets/latentqa_datasets/train/stimulus.json",
            control="datasets/latentqa_datasets/train/control.json",
            qa="datasets/latentqa_datasets/train/qa.json",
        )
        ds = latentqa_loader.load_latentqa_dataset(
            paths,
            filter_prefixes=[],
            train_percent=1.0,
            add_thought_tokens=False,
            seed=self.dataset_config.seed,
        )

        self.ds = ds

        training_data = []

        for dp in ds:
            training_data.append(create_latentqa_training_datapoint(dp, tokenizer, layers))

        self.save_dataset(training_data, "train")


class Item(BaseModel):
    role: str
    content: str


class LatentQADatapoint(BaseModel):
    label: str
    source: Literal["stimulus", "stimulus_completion", "control"]
    read_prompt: list[Item]
    dialog: list[Item]
    mask_type: str


def create_latentqa_training_datapoint(
    datapoint_dict: dict, tokenizer: AutoTokenizer, act_layers: list[int]
) -> TrainingDataPoint:
    masked_turn_count = {"stimulus_completion": 2, "stimulus": 2, "control": 0}

    datapoint = LatentQADatapoint.model_validate(datapoint_dict, strict=True)

    masked_turns = datapoint.read_prompt[: masked_turn_count[datapoint.source]]

    masked_str = tokenizer.apply_chat_template(masked_turns, tokenize=False)
    masked_tokens = tokenizer(masked_str, return_tensors=None, add_special_tokens=False, padding=False)["input_ids"]

    if datapoint.source == "stimulus_completion":
        add_generation_prompt = False
    else:
        add_generation_prompt = True

    full_read_str = tokenizer.apply_chat_template(
        datapoint.read_prompt, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    context_input_ids = tokenizer(full_read_str, return_tensors=None, add_special_tokens=False, padding=False)[
        "input_ids"
    ]

    context_positions = list(range(len(context_input_ids)))
    context_positions = context_positions[len(masked_tokens) :]

    layer = random.choice(act_layers)

    training_datapoint = create_training_datapoint(
        datapoint_type=f"latentqa_{datapoint.source}",
        prompt=datapoint.dialog[0].content,
        target_response=datapoint.dialog[1].content,
        layer=layer,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=None,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
    )

    return training_datapoint


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    config = DatasetLoaderConfig(LatentQADatasetConfig(), 100_000, 0, ["train"], model_name, [50], False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(config)
    # %%
    dataset = LatentQADatasetLoader(config)

    # %%

    dataset.create_dataset()

    print(dataset.ds[0])
    # %%
    datapoint = create_latentqa_training_datapoint(dataset.ds[0], tokenizer, [18])
    # %%
    print(tokenizer.decode(datapoint.context_input_ids))
    print(f"\n\nCTX:{tokenizer.decode(datapoint.context_input_ids[len(datapoint.context_positions) :])}")
