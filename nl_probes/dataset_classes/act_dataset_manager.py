import os
import pickle
from dataclasses import dataclass
from typing import Literal

from nl_probes.utils.dataset_utils import TrainingDataPoint


@dataclass
class BaseDatasetConfig:
    model_name: str
    layer_percents: list[int]
    seed: int
    save_acts: bool


@dataclass
class DatasetLoaderConfig:
    dataset_params: BaseDatasetConfig
    dataset_folder: str
    num_train: int
    num_test: int
    splits: list[str]
    dataset_name: str = ""


class ActDatasetLoader:
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        self.valid_splits = set(["train", "test"])
        self.dataset_config = dataset_config

        for split in self.dataset_config.splits:
            assert split in self.valid_splits, f"Invalid split: {split}"

    def create_dataset(self) -> None:
        """
        Note: Will always make all split(s) at the same time.
        This is so we ensure that train / test splits have no overlap.
        """
        raise NotImplementedError

    def load_dataset(
        self,
        split: Literal["train", "test"],
    ) -> list[TrainingDataPoint]:
        assert split in self.valid_splits, f"Invalid split: {split}"

        dataset_name = self.get_dataset_filename(split)
        filepath = os.path.join(self.dataset_config.dataset_folder, dataset_name)
        if not os.path.exists(filepath):
            os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
            self.create_dataset()
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} datapoints from {filepath}")
        return data

    def get_dataset_filename(self, split: Literal["train", "test"]) -> str:
        raise NotImplementedError
