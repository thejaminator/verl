import math
import os
import pickle
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.dataset_classes.classification_dataset_manager as classification_dataset_manager
from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook, get_introspection_prefix
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import (
    assert_no_peft_present,
    layer_percent_to_layer,
    load_model,
    load_tokenizer,
    set_seed,
)
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    create_training_datapoint,
)


@dataclass
class ClassificationDatasetConfig(BaseDatasetConfig):
    classification_dataset_name: str
    num_qa_per_sample: int = 3
    batch_size: int = 128
    end_offset: int = -3
    max_window_size: int = 20


class ClassificationDatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)

        self.dataset_params: ClassificationDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Classification dataset name gets overridden here"

        self.dataset_config.dataset_name = f"classification_{self.dataset_params.classification_dataset_name}"

        self.act_layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        assert self.dataset_params.end_offset < 0, "Offset must be negative"
        assert self.dataset_params.max_window_size > 0, "Max window size must be positive"

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        train_datapoints, test_datapoints = get_classification_datapoints(
            self.dataset_params.classification_dataset_name,
            self.dataset_params.num_qa_per_sample,
            self.dataset_config.num_train,
            self.dataset_config.num_test,
            self.dataset_config.seed,
        )

        for split in self.dataset_config.splits:
            if split == "train":
                datapoints = train_datapoints
                save_acts = self.dataset_config.save_acts
            else:
                datapoints = test_datapoints
                save_acts = True

            data = create_vector_dataset(
                datapoints,
                tokenizer,
                self.dataset_config.model_name,
                self.dataset_params.batch_size,
                self.act_layers,
                end_offset=self.dataset_params.end_offset,
                max_window_size=self.dataset_params.max_window_size,
                save_acts=save_acts,
                datapoint_type=self.dataset_config.dataset_name,
                debug_print=False,
            )

            self.save_dataset(data, split)


class ClassificationDatapoint(BaseModel):
    activation_prompt: str
    classification_prompt: str
    target_response: str


def get_classification_datapoints_from_context_qa_examples(
    examples: list[classification_dataset_manager.ContextQASample],
) -> list[ClassificationDatapoint]:
    datapoints = []
    for example in examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes' or 'No' only. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
            )
            datapoints.append(datapoint)

    return datapoints


def get_classification_datapoints(
    dataset_name: str,
    num_qa_per_sample: int,
    train_examples: int,
    test_examples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint]]:
    set_seed(random_seed)
    all_examples = classification_dataset_manager.get_samples_from_groups(
        [dataset_name],
        num_qa_per_sample,
    )

    random.shuffle(all_examples)

    assert len(all_examples) >= train_examples + test_examples, "Not enough examples to split"
    train_examples = all_examples[:train_examples]
    test_examples = all_examples[-test_examples:]

    train_datapoints = get_classification_datapoints_from_context_qa_examples(train_examples)
    test_datapoints = get_classification_datapoints_from_context_qa_examples(test_examples)

    return train_datapoints, test_datapoints


def view_tokens(tokens_L: list[int], tokenizer: AutoTokenizer, offset: int) -> None:
    print(f"Full tokens: {tokenizer.decode(tokens_L)}")
    for i in range(offset - 5, offset + 5):
        if i < len(tokens_L):
            if i == offset:
                print(f"Act token: {tokenizer.decode(tokens_L[i])}")
            else:
                print(f"Token {i}: {tokenizer.decode(tokens_L[i])}")


def create_vector_dataset(
    datapoints: list[ClassificationDatapoint],
    tokenizer: AutoTokenizer,
    model_name: str,
    batch_size: int,
    act_layers: list[int],
    end_offset: int,
    max_window_size: int,
    save_acts: bool,
    datapoint_type: str,
    debug_print: bool = False,
) -> list[TrainingDataPoint]:
    training_data = []

    assert tokenizer.padding_side == "left", "Padding side must be left"
    device = torch.device("cpu")

    if save_acts:
        model = load_model(model_name, torch.bfloat16)
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = model.device

    for i in tqdm(range(0, len(datapoints), batch_size), desc="Collecting activations"):
        batch_datapoints = datapoints[i : i + batch_size]
        formatted_prompts = []
        for datapoint in batch_datapoints:
            formatted_prompts.append([{"role": "user", "content": datapoint.activation_prompt}])
        tokenized_prompts = tokenizer.apply_chat_template(formatted_prompts, tokenize=False)
        tokenized_prompts = tokenizer(
            tokenized_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        if save_acts:
            acts_BLD_by_layer_dict = collect_activations_multiple_layers(
                model, submodules, tokenized_prompts, None, None
            )
            for layer in acts_BLD_by_layer_dict.keys():
                acts_BLD_by_layer_dict[layer] = acts_BLD_by_layer_dict[layer].to("cpu", non_blocking=True)

        tokenized_prompts["input_ids"] = tokenized_prompts["input_ids"].cpu()
        tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"].cpu()

        for layer in act_layers:
            for j in range(len(batch_datapoints)):
                attn_mask_L = tokenized_prompts["attention_mask"][j].bool()
                input_ids_L = tokenized_prompts["input_ids"][j, attn_mask_L]
                L = len(input_ids_L)
                end_pos = L + end_offset

                assert L > 0, f"L={L}"
                assert end_pos > 0, f"end_pos={end_pos}"

                k = random.randint(1, max_window_size)
                k = min(k, end_pos + 1)
                assert k > 0, f"k={k}"
                begin_pos = end_pos - k + 1
                positions_K = list(range(begin_pos, end_pos + 1))
                assert len(positions_K) == k

                # assert tokenized_prompts["input_ids"][j][offset + 1] == tokenizer.eos_token_id
                if debug_print:
                    view_tokens(input_ids_L, tokenizer, positions_K[-1])
                classification_prompt = f"{batch_datapoints[j].classification_prompt}"

                if save_acts is False:
                    acts_KD = None
                    context_input_ids = input_ids_L
                    context_positions = positions_K
                else:
                    acts_LD = acts_BLD_by_layer_dict[layer][j, attn_mask_L]
                    acts_KD = acts_LD[positions_K]
                    assert acts_KD.shape[0] == k
                    context_input_ids = None
                    context_positions = None

                training_data_point = create_training_datapoint(
                    datapoint_type=datapoint_type,
                    prompt=classification_prompt,
                    target_response=batch_datapoints[j].target_response,
                    layer=layer,
                    num_positions=k,
                    tokenizer=tokenizer,
                    acts_BD=acts_KD,
                    feature_idx=-1,
                    context_input_ids=context_input_ids,
                    context_positions=context_positions,
                )
                if training_data_point is None:
                    continue
                training_data.append(training_data_point)

    return training_data


def get_prompt_tokens_only(
    training_data_point: TrainingDataPoint,
) -> TrainingDataPoint:
    """User prompt should be labeled as -100"""
    prompt_tokens = []
    prompt_labels = []

    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(training_data_point.input_ids[i])
            prompt_labels.append(training_data_point.labels[i])
    new = training_data_point.model_copy()
    new.input_ids = prompt_tokens
    new.labels = prompt_labels
    return new


def run_classification(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    lora_path: str | None,
    hook_layer: int,
    datapoints: list[TrainingDataPoint],
    batch_size: int,
    device: torch.device,
    steering_coefficient: float,
    dtype: torch.dtype,
    generation_kwargs: dict,
):
    if lora_path is not None:
        adapter_name = lora_path
        model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    # else:

    steering_submodule = get_hf_submodule(model, hook_layer)

    results = []

    for i in range(0, len(datapoints), batch_size):
        batch_datapoints = deepcopy(datapoints[i : i + batch_size])

        for j in range(len(batch_datapoints)):
            batch_datapoints[j] = get_prompt_tokens_only(batch_datapoints[j])

        batch = construct_batch(batch_datapoints, tokenizer, device)

        hook_fn = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors,
            positions=batch.positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
        )

        tokenized_input = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }

        with add_hook(steering_submodule, hook_fn):
            outputs = model.generate(**tokenized_input, **generation_kwargs)

        response_tokens = outputs[:, batch.input_ids.shape[1] :]
        all_responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)

        for j in range(len(batch_datapoints)):
            response = all_responses[j]
            target_response = batch_datapoints[j].target_output
            results.append(
                {
                    "response": response,
                    "target_response": target_response,
                }
            )

    if lora_path is not None:
        model.delete_adapter(lora_path)

    return results


def parse_answer(answer: str) -> str:
    answer = answer.split(" ")[0]
    return answer.rstrip(".!?,;:").strip().lower()


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """
    Compute proportion statistics.

    Returns (p, se, lower, upper)
    - p: proportion correct (in [0,1])
    - se: standard error of the proportion (sqrt(p*(1-p)/n))
    - lower, upper: normal-approximation confidence interval (clamped to [0,1])

    Uses normal approx: CI = p +/- z * se. Default z=1.96 gives ~95% CI.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def analyze_results(results: list[dict]) -> dict[str, float]:
    clean_responses = []

    correct = 0
    is_correct_list = []
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        target_response = result["target_response"].lower()
        is_correct = target_response == cleaned_response
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(target_response)
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p * 100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower * 100:.2f}%, {upper * 100:.2f}%)")
    print(f"len(set(clean_responses))={len(set(clean_responses))}")

    # return values in case you want to plot programmatically
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }


if __name__ == "__main__":
    classification_datasets = {
        "geometry_of_truth": 6_000,
        "relations": 6_000,
        "sst2": 6_000,
        # "md_gender": 6_000,
        # "snli": 6_000,
        "ag_news": 6_000,
        # "ner": 6_000,
        # "tense": 6_000,
        # "language_identification": 6_000,
        # "singular_plural": 10,
    }

    all_eval_data = {}

    model_name = "Qwen/Qwen3-8B"
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokenizer = load_tokenizer(model_name)

    for dataset_name in classification_datasets.keys():
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
        )

        dataset_config = DatasetLoaderConfig(
            custom_dataset_params=classification_config,
            num_train=classification_datasets[dataset_name],
            num_test=250,
            splits=["train", "test"],
            model_name=model_name,
            layer_percents=[25, 50, 75],
            save_acts=True,
        )

        classification_dataset_loader = ClassificationDatasetLoader(
            dataset_config=dataset_config,
        )
        classification_dataset_loader.create_dataset()

#     # %%\
#     batch_size = 25
#     steering_coefficient = 2.0
#     dtype = torch.bfloat16
#     device = torch.device("cuda")
#     generation_kwargs = {
#         "do_sample": False,
#         "temperature": 0.0,
#         "max_new_tokens": 10,
#     }

#     model_name = "Qwen/Qwen3-8B"
#     hook_layer = 1

#     # %%
#     if "model" not in globals():
#         model = load_model(model_name, dtype, load_in_8bit=False)
#     # %%

#     assert_no_peft_present(model)
#     # %%

#     first_dataset = all_eval_data["geometry_of_truth"]

#     for i in range(10):
#         print(f"tokenizer.decode(first_dataset[i].input_ids): {tokenizer.decode(first_dataset[i].input_ids)}")
#         print(f"first_dataset[i].labels: {first_dataset[i].labels}")
#         print(f"first_dataset[i].target_output: {first_dataset[i].target_output}")
#         print("-" * 100)

#     # %%

#     lora_paths_with_labels = {
#         "checkpoints_multiple_datasets_layer_1_decoder/final": "SAE + Classification",
#         "checkpoints_no_sae_multiple_datasets_layer_1_decoder/final": "Classification Only",
#         None: "Original",
#     }

#     all_results = {}
#     for dataset_name in classification_datasets:
#         eval_data = all_eval_data[dataset_name]
#         all_results[dataset_name] = {}
#         for lora_path in lora_paths_with_labels.keys():
#             assert_no_peft_present(model)
#             results = run_classification(
#                 tokenizer,
#                 model,
#                 lora_path,
#                 # None,
#                 hook_layer,
#                 eval_data,
#                 batch_size,
#                 device,
#                 steering_coefficient,
#                 dtype,
#                 generation_kwargs,
#             )

#             all_results[dataset_name][lora_path] = analyze_results(results)

#     # %%
#     from pathlib import Path
#     from typing import Any

#     import matplotlib.pyplot as plt

#     def plot_classification_results(
#         all_results: dict[str, dict[str | None, dict[str, Any]]],
#         lora_paths_with_labels: dict[str | None, str],
#         *,
#         save_dir: str | Path | None = None,
#         file_format: str = "png",
#         dpi: int = 150,
#         as_percentage: bool = True,
#         annotate: bool = True,
#     ) -> list[Path]:
#         """
#         Make a bar chart per dataset with accuracy and standard error bars.

#         Args:
#             all_results: mapping like all_results[dataset_name][lora_path] -> result dict
#                         where each result dict has keys like 'p', 'se', 'n', etc.
#             lora_paths_with_labels: maps lora_path (can be None) -> label to show on x-axis
#             save_dir: if set, figures are saved here as <dataset>.<file_format>
#             file_format: e.g. 'png' or 'pdf'
#             dpi: figure DPI when saving
#             as_percentage: show accuracy in percent if True, else 0-1
#             annotate: write value above each bar

#         Returns:
#             List of saved file paths (empty if not saving).
#         """
#         if save_dir is not None:
#             save_dir = Path(save_dir)
#             save_dir.mkdir(parents=True, exist_ok=True)

#         def _slugify(s: str) -> str:
#             return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

#         saved: list[Path] = []

#         for dataset_name, per_model in all_results.items():
#             # Respect the order given in lora_paths_with_labels, but drop missing entries
#             order = [lp for lp in lora_paths_with_labels.keys() if lp in per_model]
#             if not order:
#                 continue

#             vals: list[float] = []
#             errs: list[float] = []
#             labels: list[str] = []
#             ns: list[int | None] = []

#             for lp in order:
#                 res = per_model[lp]
#                 # Prefer provided p and se; fall back if needed
#                 p = res.get("p")
#                 if p is None and "correct" in res and "n" in res and res["n"]:
#                     p = res["correct"] / res["n"]
#                 if p is None:
#                     raise ValueError(f"Missing accuracy for {dataset_name} / {lp}")

#                 se = res.get("se")
#                 if se is None and "ci_lower" in res and "ci_upper" in res:
#                     # Infer SE from a 95% CI if provided
#                     se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
#                 if se is None:
#                     se = 0.0

#                 n = res.get("n")

#                 if as_percentage:
#                     vals.append(p * 100.0)
#                     errs.append(se * 100.0)
#                 else:
#                     vals.append(float(p))
#                     errs.append(float(se))

#                 labels.append(lora_paths_with_labels[lp])
#                 ns.append(n)

#             # One figure per dataset
#             fig = plt.figure(figsize=(6.5, 4.2))
#             ax = plt.gca()

#             x = list(range(len(order)))
#             bars = ax.bar(x, vals, yerr=errs, capsize=4)

#             xticklabels = [f"{lab}\n(n={n})" if n is not None else lab for lab, n in zip(labels, ns)]
#             ax.set_xticks(x, xticklabels, rotation=0)

#             ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
#             ax.set_title(dataset_name)
#             ax.set_ylim(0, 100 if as_percentage else 1.0)
#             ax.yaxis.grid(True, linestyle="--", alpha=0.4)

#             if annotate:
#                 for b, v in zip(bars, vals):
#                     ax.text(
#                         b.get_x() + b.get_width() / 2.0,
#                         v,
#                         f"{v:.1f}" + ("%" if as_percentage else ""),
#                         ha="center",
#                         va="bottom",
#                         fontsize=9,
#                     )

#             fig.tight_layout()

#             if save_dir is not None:
#                 out_path = save_dir / f"{_slugify(dataset_name)}.{file_format}"
#                 fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
#                 saved.append(out_path)

#             plt.show()
#             plt.close(fig)

#         return saved

#     plot_classification_results(all_results, lora_paths_with_labels, save_dir=None)

#     # %%
