# Copyright 2025 Individual Contributor: Mert Unsal
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

import torch

from feature_vector_reward import REWARD_CALLER
from feature_vector_reward import compute_score as compute_score_feature_vector
from verl import DataProto
from verl.workers.reward_manager import register


@register("batch_detection")
class BatchDetectionRewardManager:
    """
    Modfied batch reward manager that computes detection scores.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", {})
        sae = data.non_tensor_batch["sae"]

        try:
            scores = compute_score_feature_vector(
                data_source=data_sources,
                solution_str=responses_str,
                sae=sae,
                extra_info=extras,
            )
        except Exception as e:
            breakpoint()
            raise e

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Ensure the caller cache is reloaded at the start
        # Because damned verl wants to do a seperate ray process for everything, the cache between
        # our fire and forget reward computation and the main process gets out of sync.
        REWARD_CALLER.reload_file_cache()

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}
        table_data = []

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]

            # Always collect table data for logging (separate from printing logic)
            response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
            table_data.append(
                {
                    "sae": data.non_tensor_batch["sae"][i]["sae_id"],
                    "layer": data.non_tensor_batch["sae"][i]["sae_info"]["sae_layer"],
                    "explanation": response_str,
                    "score": scores[i],
                }
            )

            # Only print if within num_examine limit
            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                # ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # Add table data to reward_extra_info for logging (like NaiveRewardManager does)
        # This will get automatically converted to np.array by the trainer
        if table_data:
            reward_extra_info["table_data"] = table_data

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
