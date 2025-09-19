# %%
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
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
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
    get_prompt_tokens_only,
)
from nl_probes.utils.eval import parse_answer, run_evaluation, score_eval_responses

# %%

main_test_size = 250
classification_datasets = {
    "geometry_of_truth": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "relations": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "sst2": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "md_gender": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "snli": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "ag_news": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "ner": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "tense": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    "language_identification": {
        "num_train": 0,
        "num_test": main_test_size,
        "splits": ["test"],
    },
    "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
}

lora_paths_with_labels = {
    "checkpoints_act_pretrain/final": "Pretrain Mix",
    "checkpoints_act_pretrain_posttrain/final": "Pretrain Mix -> 1 Classification Epoch",
    "checkpoints_classification_only_2_epochs/final": "2 Classification Epochs",
    None: "Original",
}

all_eval_data = {}

model_name = "Qwen/Qwen3-8B"
dtype = torch.bfloat16
device = torch.device("cuda")
tokenizer = load_tokenizer(model_name)

classification_dataset_loaders: list[ClassificationDatasetLoader] = []
layer_percents = [25, 50, 75]
batch_size = 128
steering_coefficient = 2.0
hook_layer = 1
generation_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

for dataset_name in classification_datasets.keys():
    classification_config = ClassificationDatasetConfig(
        classification_dataset_name=dataset_name,
    )

    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=classification_config,
        num_train=classification_datasets[dataset_name]["num_train"],
        num_test=classification_datasets[dataset_name]["num_test"],
        splits=classification_datasets[dataset_name]["splits"],
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=False,
    )

    classification_dataset_loader = ClassificationDatasetLoader(
        dataset_config=dataset_config,
    )
    classification_dataset_loaders.append(classification_dataset_loader)

all_eval_data: dict[str, list[TrainingDataPoint]] = {}

for dataset_loader in classification_dataset_loaders:
    if "test" in dataset_loader.dataset_config.splits:
        all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

# %%

model = load_model(model_name, dtype, load_in_8bit=False)
submodule = get_hf_submodule(model, hook_layer)
# %%
all_results = {}
for dataset_name in all_eval_data.keys():
    eval_data = all_eval_data[dataset_name]
    all_results[dataset_name] = {}
    for lora_path in lora_paths_with_labels.keys():
        results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=lora_path,
            eval_batch_size=batch_size,
            steering_coefficient=steering_coefficient,
            generation_kwargs=generation_kwargs,
        )
        all_results[dataset_name][lora_path] = results
# %%

print(all_results.keys())
first_key = list(all_results.keys())[0]
print(all_results[first_key].keys())
# print(all_results)
# %%


def format_results(results: list[FeatureResult], eval_data: list[TrainingDataPoint]) -> list[dict[str, str]]:
    formatted_results = []
    for result, eval_data_point in zip(results, eval_data, strict=True):
        cleaned_response = parse_answer(result.api_response)
        target_response = parse_answer(eval_data_point.target_output)
        formatted_results.append(
            {
                "response": cleaned_response,
                "target_response": target_response,
            }
        )
    return formatted_results


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


lora_paths_with_labels = {
    "checkpoints_act_pretrain_posttrain/final": "SAE + Classification",
    "checkpoints_classification_only_2_epochs/final": "Classification Only",
    None: "Original",
}


final_results = {}
for dataset_name in all_eval_data.keys():
    eval_data = all_eval_data[dataset_name]
    final_results[dataset_name] = {}
    for lora_path in lora_paths_with_labels.keys():
        results = all_results[dataset_name][lora_path]
        results = format_results(results, eval_data)

        final_results[dataset_name][lora_path] = analyze_results(results)


# %%

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_classification_results(
    all_results: dict[str, dict[str | None, dict[str, Any]]],
    lora_paths_with_labels: dict[str | None, str],
    iid_ds: list[str],
    ood_ds: list[str],
    *,
    save_dir: str | Path | None = None,
) -> None:
    """
    Make a bar chart per dataset with accuracy and standard error bars.

    Args:
        all_results: mapping like all_results[dataset_name][lora_path] -> result dict
                    where each result dict has keys like 'p', 'se', 'n', etc.
        lora_paths_with_labels: maps lora_path (can be None) -> label to show on x-axis
        save_dir: if set, figures are saved here as <dataset>.<file_format>
        file_format: e.g. 'png' or 'pdf'
        dpi: figure DPI when saving
        as_percentage: show accuracy in percent if True, else 0-1
        annotate: write value above each bar

    Returns:
        List of saved file paths (empty if not saving).
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Define colors and line styles for consistency
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#000000"]
    markers = ["o", "o", "o", "o"]
    linestyles = ["--", "--", "--", "--"]

    # Create first plot for IID datasets
    plt.figure(figsize=(10, 6))

    for idx, (lora_path, label) in enumerate(lora_paths_with_labels.items()):
        iid_scores = []
        for dataset in iid_ds:
            iid_scores.append(all_results[dataset][lora_path]["p"])

        plt.plot(
            range(len(iid_ds)),
            iid_scores,
            label=label,
            marker=markers[idx],
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    # Add random chance baseline
    plt.axhline(y=0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="Random Chance Baseline")

    # Customize IID plot
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("IID (In-Distribution) Dataset Performance", fontsize=14, fontweight="bold")
    plt.xticks(range(len(iid_ds)), iid_ds, rotation=45, ha="right")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([0.45, 1.0])
    plt.tight_layout()
    plt.show()

    # Create second plot for OOD datasets
    plt.figure(figsize=(10, 6))

    for idx, (lora_path, label) in enumerate(lora_paths_with_labels.items()):
        ood_scores = []
        for dataset in ood_ds:
            ood_scores.append(all_results[dataset][lora_path]["p"])

        plt.plot(
            range(len(ood_ds)),
            ood_scores,
            label=label,
            marker=markers[idx],
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    # Add random chance baseline
    plt.axhline(y=0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="Random Chance Baseline")

    # Customize OOD plot
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("OOD (Out-of-Distribution) Dataset Performance", fontsize=14, fontweight="bold")
    plt.xticks(range(len(ood_ds)), ood_ds, rotation=45, ha="right")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([0.45, 1.0])
    plt.tight_layout()
    plt.show()

    # Calculate and print average scores for each method
    print("\n=== Average Performance ===")
    for lora_path, label in lora_paths_with_labels.items():
        iid_avg = np.mean([all_results[ds][lora_path]["p"] for ds in iid_ds])
        ood_avg = np.mean([all_results[ds][lora_path]["p"] for ds in ood_ds])
        print(f"\n{label}:")
        print(f"  IID Average: {iid_avg:.4f}")
        print(f"  OOD Average: {ood_avg:.4f}")
        print(f"  Overall Average: {np.mean([iid_avg, ood_avg]):.4f}")


iid_ds = [
    "classification_geometry_of_truth",
    "classification_relations",
    "classification_sst2",
    "classification_md_gender",
    "classification_snli",
    # "classification_ag_news",
    "classification_ner",
    "classification_tense",
    # "classification_language_identification",
    # "classification_singular_plural",
]

ood_ds = [
    "classification_ag_news",
    "classification_language_identification",
    "classification_singular_plural",
]


plot_classification_results(final_results, lora_paths_with_labels, iid_ds, ood_ds, save_dir=None)

# %%
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _extract_p_se_n(res: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    """
    Return (p, se, n) in 0-1 units if available. Falls back to compute p from correct/n.
    If SE is missing but a 95% CI is given, infer SE from CI width.
    """
    p = res.get("p")
    n = res.get("n")

    if p is None and "correct" in res and "n" in res and res["n"]:
        p = res["correct"] / res["n"]

    se = res.get("se")
    if se is None and "ci_lower" in res and "ci_upper" in res:
        se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)

    if p is None:
        return None, None, n
    if se is None:
        se = 0.0

    return float(p), float(se), (int(n) if n is not None else None)


def _build_matrices(
    all_results: dict[str, dict[str | None, dict[str, Any]]],
    lora_paths_with_labels: dict[str | None, str],
    datasets: list[str],
    as_percentage: bool,
) -> tuple[list[str], list[str], list[str | None], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        ds_list: kept datasets in given order if present in all_results
        model_labels: labels aligned with order_lps
        order_lps: LoRA path keys in the order given by lora_paths_with_labels but filtered to those present
        Y: shape (M, N) values (percent or 0-1)
        E: shape (M, N) standard errors in same units
        N: shape (M, N) sample sizes (nan if missing)
    """
    ds_list = [d for d in datasets if d in all_results]
    # Only keep models that appear at least once in the chosen datasets
    present_lps = set()
    for d in ds_list:
        present_lps |= set(all_results[d].keys())
    order_lps = [lp for lp in lora_paths_with_labels.keys() if lp in present_lps]
    model_labels = [lora_paths_with_labels[lp] for lp in order_lps]

    M, N = len(order_lps), len(ds_list)
    Y = np.full((M, N), np.nan, dtype=float)
    E = np.zeros((M, N), dtype=float)
    Nmat = np.full((M, N), np.nan, dtype=float)

    scale = 100.0 if as_percentage else 1.0

    for j, ds in enumerate(ds_list):
        per_model = all_results.get(ds, {})
        for i, lp in enumerate(order_lps):
            if lp not in per_model:
                continue
            p, se, n = _extract_p_se_n(per_model[lp])
            if p is None:
                continue
            Y[i, j] = p * scale
            E[i, j] = se * scale
            if n is not None:
                Nmat[i, j] = float(n)

    return ds_list, model_labels, order_lps, Y, E, Nmat


def plot_multi_dataset_lines(
    all_results: dict[str, dict[str | None, dict[str, Any]]],
    lora_paths_with_labels: dict[str | None, str],
    datasets: list[str],
    *,
    title: str,
    as_percentage: bool = True,
    annotate: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    ds_list, model_labels, order_lps, Y, E, Nmat = _build_matrices(
        all_results, lora_paths_with_labels, datasets, as_percentage
    )
    if len(ds_list) == 0 or len(model_labels) == 0:
        return

    fig_w = max(6.5, 0.9 * len(ds_list) + 2.0)
    fig = plt.figure(figsize=(fig_w, 4.2))
    ax = plt.gca()

    x = np.arange(len(ds_list))
    for i, label in enumerate(model_labels):
        y = Y[i]
        e = E[i]
        ax.errorbar(x, y, yerr=e, fmt="o-", capsize=3, label=label)
        if annotate:
            for xi, yi in zip(x, y):
                if not np.isnan(yi):
                    ax.text(xi, yi, f"{yi:.1f}" + ("%" if as_percentage else ""), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_list, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 100 if as_percentage else 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncols=min(3, len(model_labels)), frameon=False)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_multi_dataset_grouped_bars(
    all_results: dict[str, dict[str | None, dict[str, Any]]],
    lora_paths_with_labels: dict[str | None, str],
    datasets: list[str],
    *,
    title: str,
    as_percentage: bool = True,
    annotate: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    ds_list, model_labels, order_lps, Y, E, Nmat = _build_matrices(
        all_results, lora_paths_with_labels, datasets, as_percentage
    )
    if len(ds_list) == 0 or len(model_labels) == 0:
        return

    M, N = Y.shape
    fig_w = max(6.5, 0.9 * N + 2.5)
    fig = plt.figure(figsize=(fig_w, 4.2))
    ax = plt.gca()

    x = np.arange(N)
    width = 0.82 / max(1, M)
    offsets = (np.arange(M) - (M - 1) / 2.0) * width

    bars_all = []
    for i, label in enumerate(model_labels):
        xi = x + offsets[i]
        bars = ax.bar(xi, Y[i], yerr=E[i], width=width, capsize=3, label=label)
        bars_all.append(bars)
        if annotate:
            for b, yi in zip(bars, Y[i]):
                if not np.isnan(yi):
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        yi,
                        f"{yi:.1f}" + ("%" if as_percentage else ""),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    xticklabels = []
    for j, ds in enumerate(ds_list):
        # show n from the first model that has it
        n_any = None
        for i in range(M):
            val = Nmat[i, j]
            if not np.isnan(val):
                n_any = int(val)
                break
        xticklabels.append(f"{ds}\n(n={n_any})" if n_any is not None else ds)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 100 if as_percentage else 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncols=min(3, len(model_labels)), frameon=False)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_group_summary_averages(
    all_results: dict[str, dict[str | None, dict[str, Any]]],
    lora_paths_with_labels: dict[str | None, str],
    datasets: list[str],
    *,
    title_suffix: str,
    as_percentage: bool = True,
    save_dir: str | Path | None = None,
    dpi: int = 150,
):
    """
    Makes two figures:
      1) Micro-average bars per model (weighted by n). Error bar uses binomial SE over total N.
      2) Macro-average bars per model (mean across datasets). Error bar is SE of the mean across datasets.
    """
    ds_list, model_labels, order_lps, Y, E, Nmat = _build_matrices(
        all_results, lora_paths_with_labels, datasets, as_percentage
    )
    if len(ds_list) == 0 or len(model_labels) == 0:
        return

    scale = 100.0 if as_percentage else 1.0

    # ---------- Micro-average ----------
    p_micro = []
    se_micro = []
    for i in range(len(model_labels)):
        vals = Y[i] / scale  # back to 0-1
        ns = Nmat[i]
        mask = ~np.isnan(vals) & ~np.isnan(ns) & (ns > 0)
        if not np.any(mask):
            p_micro.append(np.nan)
            se_micro.append(0.0)
            continue
        total_n = float(np.nansum(ns[mask]))
        total_correct = float(np.nansum(vals[mask] * ns[mask]))
        p_hat = total_correct / total_n
        # Binomial SE
        se_hat = math.sqrt(max(1e-12, p_hat * (1 - p_hat) / total_n))
        p_micro.append(p_hat * scale)
        se_micro.append(se_hat * scale)

    # ---------- Macro-average ----------
    p_macro = []
    se_macro = []
    for i in range(len(model_labels)):
        vals = Y[i]  # already in chosen units
        mask = ~np.isnan(vals)
        if not np.any(mask):
            p_macro.append(np.nan)
            se_macro.append(0.0)
            continue
        mean = float(np.nanmean(vals[mask]))
        std = float(np.nanstd(vals[mask], ddof=1)) if np.sum(mask) > 1 else 0.0
        se = std / math.sqrt(max(1, int(np.sum(mask))))
        p_macro.append(mean)
        se_macro.append(se)

    def _plot_summary(vals: list[float], errs: list[float], title: str, fname: str | None):
        fig = plt.figure(figsize=(max(6.0, 1.2 * len(model_labels)), 3.8))
        ax = plt.gca()
        x = np.arange(len(model_labels))
        bars = ax.bar(x, vals, yerr=errs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=0)
        ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 100 if as_percentage else 1.0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    v,
                    f"{v:.1f}" + ("%" if as_percentage else ""),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.tight_layout()
        if fname is not None:
            p = Path(fname)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    save_dir = Path(save_dir) if save_dir is not None else None
    micro_name = (save_dir / f"summary_micro_{title_suffix}.png") if save_dir is not None else None
    macro_name = (save_dir / f"summary_macro_{title_suffix}.png") if save_dir is not None else None

    _plot_summary(
        p_micro, se_micro, f"Micro-average accuracy - {title_suffix}", str(micro_name) if micro_name else None
    )
    _plot_summary(
        p_macro, se_macro, f"Macro-average accuracy - {title_suffix}", str(macro_name) if macro_name else None
    )


plot_multi_dataset_lines(
    final_results,
    lora_paths_with_labels,
    iid_ds,
    title="IID datasets",
    as_percentage=True,
    annotate=False,
    save_path=None,
)
plot_multi_dataset_lines(
    final_results,
    lora_paths_with_labels,
    ood_ds,
    title="OOD datasets",
    as_percentage=True,
    annotate=False,
    save_path=None,
)
# %%
# If you prefer grouped bars instead of lines
plot_multi_dataset_grouped_bars(all_results, lora_paths_with_labels, iid_ds, title="IID datasets")
plot_multi_dataset_grouped_bars(all_results, lora_paths_with_labels, ood_ds, title="OOD datasets")

# Optional summaries
plot_group_summary_averages(
    all_results, lora_paths_with_labels, iid_ds, title_suffix="IID", as_percentage=True, save_dir=None
)
plot_group_summary_averages(
    all_results, lora_paths_with_labels, ood_ds, title_suffix="OOD", as_percentage=True, save_dir=None
)
# %%
results_filename = "0919_classification_results_multiple_datasets_layer_1_decoder.json"
import json

with open(results_filename, "w") as f:
    json.dump(all_results, f)
# %%
