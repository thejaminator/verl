# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Mapping

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # First pass: collect all response lengths grouped by prompt_ids
        prompt_to_lengths: dict[tuple[int], list[int]] = defaultdict(list)
        prompt_id_to_key = {}  # Map from data index to prompt key
        corrects: list[bool] = []

        # Collect data for logging table - one entry per batch item
        table_data: list[dict] = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]

            prompt_length = prompt_ids.shape[-1]
            # 0s after EOS token
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # Use prompt_ids as key (convert to tuple for hashing)
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_key = tuple(valid_prompt_ids.tolist())

            prompt_to_lengths[prompt_key].append(int(valid_response_length))
            prompt_id_to_key[i] = prompt_key

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            if i == 0:
                print("[prompt]" + prompt_str + "\n")
                print("[response]" + response_str + "\n")
                # lengths for this prompt
                prompt_key = prompt_id_to_key[i]
                print("[lengths]" + str(prompt_to_lengths[prompt_key]) + "\n")
                print("[response_length]" + str(valid_response_length) + "\n")
                print("[ground_truth]" + ground_truth + "\n")

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # Add all_lengths for this prompt group to extra_info
            prompt_key = prompt_id_to_key[i]
            extra_info["all_lengths"] = prompt_to_lengths[prompt_key]

            # Add response_length for this prompt group to extra_info
            extra_info["response_length"] = valid_response_length

            score: Mapping[str, str | float | bool] = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )  # type: ignore

            assert "score" in score, "score must contain 'score' key, reward function should add it"
            assert "is_correct" in score, "score must contain 'is_correct' key, reward function should add it"
            assert "parsed_answer" in score, "score must contain 'parsed_answer' key, reward function should add it"
            parsed_answer: str = score["parsed_answer"]  # type: ignore
            is_correct: bool = score["is_correct"]  # type: ignore
            corrects.append(is_correct)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # Collect data for table logging (will be passed through reward_extra_info)
            if isinstance(score, dict):
                score_value = float(score["score"])  # type: ignore
            else:
                score_value = float(score)
            table_data.append(
                {
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "length": valid_response_length,
                    "is_correct": is_correct,
                    "parsed_answer": parsed_answer,
                    "score": score_value,
                }
            )

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        # Add table data and batch accuracy to reward_extra_info for logging in training loop
        reward_extra_info["table_data"] = table_data  # List of individual row dicts
        reward_extra_info["batch_accuracy"] = corrects  # List of booleans per batch item

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
