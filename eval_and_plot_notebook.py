# %%
"""
Clean, single-file, notebook-style evaluation and plotting script.

Key improvements:
- Canonical method keys (strings only) -> JSON-safe round-trips
- Dataset name canonicalization (strip 'classification_' prefix)
- Robust JSON load/save + migration from old shapes (including 'null' keys)
- Plotting that tolerates missing datasets/methods
- Minimal, readable, hackable helpers

Python: 3.11
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# External project imports (assumed available in your env)
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import parse_answer, run_evaluation

# -----------------------------
# Configuration - tune here
# -----------------------------

RESULTS_FILENAME = "0919_classification_results.json"

# When False: skip expensive eval and try to load from RESULTS_FILENAME
RUN_FRESH_EVAL = True

# Model and eval config
MODEL_NAME = "Qwen/Qwen3-8B"
HOOK_LAYER = 1
DTYPE = torch.bfloat16
BATCH_SIZE = 128
STEERING_COEFFICIENT = 2.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

# Dataset selection
MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "relations": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "sst2": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "md_gender": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "snli": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ag_news": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "tense": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "language_identification": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
}

# Groupings for plotting
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]
OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
]

# Layer percent settings used by loaders
LAYER_PERCENTS = [25, 50, 75]

KEY_FOR_NONE = "original"


@dataclass(frozen=True)
class Method:
    label: str
    lora_path: str


METHODS: list[Method] = [
    Method(label="Original", lora_path=KEY_FOR_NONE),
    Method(
        label="Past Lens Pretrain Mix -> 1 Classification Epoch",
        lora_path="checkpoints_act_only_1_token_-3_-5_classification_posttrain/final",
    ),
    Method(
        label="SAE Pretrain Mix -> 1 Classification Epoch",
        lora_path="checkpoints_sae_pretrain_1_token_-3_-5_classification_posttrain/final",
    ),
    Method(
        label="SAE and Past Lens Pretrain Mix -> 1 Classification Epoch",
        lora_path="checkpoints_all_pretrain_1_token_-3_-5_sae_explanation_posttrain/final",
    ),
    Method(
        label="2 Classification Epochs",
        lora_path="checkpoints_classification_only_1_token_-3_-5_2_epochs/final",
    ),
]

LABEL_BY_LORA_PATH: dict[str, str] = {m.lora_path: m.label for m in METHODS}

# Optional external baselines keyed by dataset id; set to None to skip plotting.
LINEAR_PROBE_BASELINE: dict[str, Any] | None = {
    "label": "Linear Probe Baseline",
    "values": {
        "ag_news": 0.8338,
        "geometry_of_truth": 0.8760,
        "language_identification": 0.8862,
        "md_gender": 0.9382,
        "singular_plural": 0.9618,
        "sst2": 0.8227,
        "tense": 0.9484,
    },
}
# -----------------------------
# Lightweight helpers
# -----------------------------


def canonical_dataset_id(name: str) -> str:
    """Strip 'classification_' prefix if present so keys match your IID/OOD lists."""
    if name.startswith("classification_"):
        return name[len("classification_") :]
    return name


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """Return p, se, lower, upper for a binomial proportion with normal approx CI."""
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def score_predictions(cleaned_responses: list[str], target_responses: list[str]) -> dict[str, Any]:
    """Compute correctness stats given cleaned model outputs and cleaned targets."""
    assert len(cleaned_responses) == len(target_responses)
    n = len(cleaned_responses)
    is_correct_list = [cr == tr for cr, tr in zip(cleaned_responses, target_responses)]
    correct = sum(is_correct_list)
    p, se, lower, upper = proportion_confidence(correct, n)
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }


# %%
# Tokenizer and dataset loading

tokenizer = load_tokenizer(MODEL_NAME)

classification_dataset_loaders: list[ClassificationDatasetLoader] = []
for dataset_name, dcfg in CLASSIFICATION_DATASETS.items():
    classification_config = ClassificationDatasetConfig(
        classification_dataset_name=dataset_name,
        max_end_offset=-5,
        min_end_offset=-3,
        max_window_size=1,
    )
    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=classification_config,
        num_train=dcfg["num_train"],
        num_test=dcfg["num_test"],
        splits=dcfg["splits"],
        model_name=MODEL_NAME,
        layer_percents=LAYER_PERCENTS,
        save_acts=False,
    )
    classification_dataset_loaders.append(ClassificationDatasetLoader(dataset_config=dataset_config))

# Pull test sets for evaluation
all_eval_data: dict[str, list[Any]] = {}
for loader in classification_dataset_loaders:
    if "test" in loader.dataset_config.splits:
        ds_id = canonical_dataset_id(loader.dataset_config.dataset_name)
        all_eval_data[ds_id] = loader.load_dataset("test")

print(f"Loaded datasets: {list(all_eval_data.keys())}")

# %%
# Model and submodule

device = torch.device("cuda")
dtype = torch.bfloat16
print(f"Using device={device}, dtype={dtype}")

model = load_model(MODEL_NAME, dtype, load_in_8bit=False)
submodule = get_hf_submodule(model, HOOK_LAYER)

# %%
# Evaluation (fast path: load JSON if available, heavy path: run fresh)


def run_eval_for_datasets(eval_data_by_ds: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """
    Returns:
        results[dataset_id][method_key] -> metrics dict
    """
    out: dict[str, dict[str, Any]] = {}
    for ds_id, eval_data in eval_data_by_ds.items():
        out[ds_id] = {}
        for m in METHODS:
            # we do this as None is not convenient to store in JSON
            if m.lora_path == KEY_FOR_NONE:
                lora_path = None
            else:
                lora_path = m.lora_path
            # Heavy call - returns list of FeatureResult-like with .api_response
            raw_results = run_evaluation(
                eval_data=eval_data,
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                device=device,
                dtype=dtype,
                global_step=-1,
                lora_path=lora_path,
                eval_batch_size=BATCH_SIZE,
                steering_coefficient=STEERING_COEFFICIENT,
                generation_kwargs=GENERATION_KWARGS,
            )

            cleaned = [parse_answer(r.api_response) for r in raw_results]
            targets = [parse_answer(dp.target_output) for dp in eval_data]
            metrics = score_predictions(cleaned, targets)
            out[ds_id][m.lora_path] = metrics
            print(f"[{ds_id}] {m.label}: p={metrics['p']:.3f} n={metrics['n']}")
    return out


# Orchestrate load-or-run
results_by_ds: dict[str, dict[str, Any]] = {}
meta_loaded: dict[str, Any] | None = None

if not RUN_FRESH_EVAL and Path(RESULTS_FILENAME).exists():
    print(f"Loading existing results from {RESULTS_FILENAME}")
    with open(RESULTS_FILENAME, "r") as f:
        results_by_ds = json.load(f)
else:
    print("Running fresh evaluation - this can be slow.")
    results_by_ds = run_eval_for_datasets(all_eval_data)
    with open(RESULTS_FILENAME, "w") as f:
        json.dump(results_by_ds, f)

# %%
# Plotting utilities - robust to missing entries


def _score_and_err(result: dict[str, Any] | None) -> tuple[float, float, float]:
    """Return (p, lower_err, upper_err). If missing, return (nan,0,0)."""
    if not result:
        return float("nan"), 0.0, 0.0
    p = result.get("p")
    if p is None:
        n = result.get("n", 0) or 0
        c = result.get("correct", 0) or 0
        p = (c / n) if n else float("nan")
    lower = result.get("ci_lower")
    upper = result.get("ci_upper")
    if lower is not None and upper is not None and isinstance(p, (int, float)):
        return float(p), max(0.0, float(p) - float(lower)), max(0.0, float(upper) - float(p))
    se = result.get("se")
    if se is not None and isinstance(p, (int, float)):
        return float(p), float(se), float(se)
    return float(p) if isinstance(p, (int, float)) else float("nan"), 0.0, 0.0


def plot_group(
    group_name: str,
    datasets: list[str],
    results: dict[str, dict[str, Any]],
    label_by_key: dict[str, str],
    *,
    baseline: float | None = 0.5,
    extra_series: list[dict[str, Any]] | None = None,
) -> None:
    present = [ds for ds in datasets if ds in results]
    missing = [ds for ds in datasets if ds not in results]
    if missing:
        print(f"[plot {group_name}] Skipping missing datasets: {missing}")

    if not present:
        print(f"[plot {group_name}] Nothing to plot.")
        return

    x = np.arange(len(present))
    plt.figure(figsize=(10, 6))

    # For consistent legend order, iterate methods in METHODS order but include only those present
    for m in METHODS:
        y = []
        yerr_low = []
        yerr_high = []
        any_present = False
        for ds in present:
            r = results[ds].get(m.lora_path)
            p, el, eh = _score_and_err(r)
            if not np.isnan(p):
                any_present = True
            y.append(p)
            yerr_low.append(el)
            yerr_high.append(eh)
        if any_present:
            # Build legend label with per-plot average over non-NaN values
            base_label = label_by_key.get(m.lora_path, m.lora_path)
            if len(base_label) > 22:
                base_label = base_label.replace(" -> ", "->\n ")
            valid = [v for v in y if not np.isnan(v)]
            label = f"{base_label} (avg={np.mean(valid):.3f})" if valid else base_label

            plt.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                label=label,
                marker="o",
                linestyle="--",
                linewidth=2,
                markersize=6,
                alpha=0.9,
                capsize=4,
            )

    if extra_series:
        for series in extra_series:
            values = series.get("values") or series.get("data") or {}
            if not isinstance(values, dict):
                continue
            y = []
            any_present = False
            for ds in present:
                val = values.get(ds)
                if val is None:
                    y.append(np.nan)
                else:
                    any_present = True
                    y.append(float(val))
            if not any_present:
                continue
            label = series.get("label", "Extra")
            valid = [v for v in y if not np.isnan(v)]
            display_label = f"{label} (avg={np.mean(valid):.3f})" if valid else label
            plt.plot(
                x,
                y,
                marker="s",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                alpha=0.8,
                label=display_label,
            )

    if baseline is not None:
        plt.axhline(y=baseline, linestyle=":", linewidth=1.5, alpha=0.7, label=f"Baseline {baseline:.2f}")

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title(group_name)
    plt.xticks(x, present, rotation=45, ha="right")
    # plt.legend(loc="best", fontsize=10)
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc="center", ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([0.45, 1.0])
    plt.tight_layout()
    plt.show()


# Plot IID and OOD

plot_group(
    "IID (In-Distribution) Dataset Performance, Single Token",
    IID_DATASETS,
    results_by_ds,
    LABEL_BY_LORA_PATH,
    baseline=0.5,
    extra_series=[LINEAR_PROBE_BASELINE] if LINEAR_PROBE_BASELINE else None,
)
plot_group(
    "OOD (Out-of-Distribution) Dataset Performance, Single Token",
    OOD_DATASETS,
    results_by_ds,
    LABEL_BY_LORA_PATH,
    baseline=0.5,
    extra_series=[LINEAR_PROBE_BASELINE] if LINEAR_PROBE_BASELINE else None,
)
# %%
# Inspect a single dataset's per-example correctness, if you have raw predictions in-memory.
# This cell is optional - it shows how you could re-run scoring on a dataset interactively.


def analyze_results_debug(
    eval_data: list[Any],
    raw_api_responses: list[str],
) -> dict[str, Any]:
    """
    Convenience to debug a single method on a single dataset.
    Prints first few mismatches and returns stats.
    """
    cleaned = [parse_answer(x) for x in raw_api_responses]
    targets = [parse_answer(dp.target_output) for dp in eval_data]

    stats = score_predictions(cleaned, targets)

    # Print a handful of mismatches for quick inspection
    mismatches = [(i, c, t) for i, (c, t, ok) in enumerate(zip(cleaned, targets, stats["is_correct_list"])) if not ok][
        :10
    ]
    if mismatches:
        print("\nFirst few mismatches (index, response, target):")
        for i, c, t in mismatches:
            print(f"{i:4d}: {c!r} vs {t!r}")

    print(
        f"\ncorrect={stats['correct']}, n={stats['n']}, p={stats['p']:.4f} "
        f"95%CI=[{stats['ci_lower']:.4f},{stats['ci_upper']:.4f}]"
    )
    return stats


# Example usage (uncomment to run live on one dataset and one method):
# ds = "sst2"
# method = METHODS[0]  # Original
# fresh_raw = run_evaluation(
#     eval_data=all_eval_data[ds],
#     model=model,
#     tokenizer=tokenizer,
#     submodule=submodule,
#     device=device,
#     dtype=dtype,
#     global_step=-1,
#     lora_path=method.lora_path,
#     eval_batch_size=BATCH_SIZE,
#     steering_coefficient=STEERING_COEFFICIENT,
#     generation_kwargs=GENERATION_KWARGS,
# )
# analyze_results_debug(all_eval_data[ds], [r.api_response for r in fresh_raw])

# %%
