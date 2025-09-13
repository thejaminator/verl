import os
import pickle

from sft_config import TrainingDataPoint


class ActDatasetLoader:
    def __init__(
        self, dataset_name: str, dataset_params: dict, dataset_folder: str, num_train: int | None, num_test: int | None
    ):
        """
        If num_train is None, all examples will be used for training.
        If num_test is None, all examples will be used for testing.
        If num_train == 0, no examples will be used for training.
        If num_test == 0, no examples will be used for testing.
        """

        assert num_train or num_test, "Either num_train or num_test must be provided"
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.dataset_folder = dataset_folder
        self.num_train = num_train
        self.num_test = num_test

    def create_dataset(self) -> list[TrainingDataPoint]:
        raise NotImplementedError

    def load_dataset(self) -> list[TrainingDataPoint]:
        test_dataset_name = self.get_dataset_filename()
        filepath = os.path.join(self.dataset_folder, test_dataset_name)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                test_data = pickle.load(f)

            print(f"Loaded {len(test_data)} datapoints from {filepath}")
            return test_data

        dataset = self.create_dataset()

        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset)} datapoints to {test_dataset_name}")

        return dataset

    def get_dataset_filename(self) -> str:
        raise NotImplementedError
