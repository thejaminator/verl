# %%

import json

filename = "personaqa_probe_results_v1.json"
filename = "personaqa_probe_results_yes_no.json"

with open(filename, "r") as f:
    data = json.load(f)


# %%
print(data.keys())
# %%
act_keys = ["lora", "orig", "diff"]

PROMPT_TYPES: list[str] = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

for prompt_type in PROMPT_TYPES:
    print(f"\n\n\n{prompt_type}")
    for key in act_keys:
        print(f"\n{key}")
        contained = []
        all_nonzero = []
        for r in data["records"]:
            if r["act_key"] == key and r["prompt_type"] == prompt_type:
                contained.append(r["mean_ground_truth_containment"])
                nonzero = 0
                if r["mean_ground_truth_containment"] > 0:
                    nonzero = 1
                all_nonzero.append(nonzero)

        mean_containment_overall = sum(contained) / len(contained)
        print(
            f"Summary - records: {len(contained)} - mean containment: {mean_containment_overall:.4f}, num_nonzero: {sum(all_nonzero)}"
        )
# %%
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
filename = "personaqa_probe_results_yes_no_v2.json"
with open(filename, "r") as f:
    data = json.load(f)

# Define constants
act_keys = ["lora", "orig", "diff"]
PROMPT_TYPES = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

# Process data to calculate metrics
metrics = {}
for prompt_type in PROMPT_TYPES:
    metrics[prompt_type] = {}
    for key in act_keys:
        contained = []
        all_nonzero = []
        for r in data["records"]:
            if r["act_key"] == key and r["prompt_type"] == prompt_type:
                containment = []

                num_tokens = len(r["token_responses"])

                for i in range(num_tokens - 9, num_tokens - 4):
                    if r["ground_truth"].lower() in r["token_responses"][i].lower():
                        containment.append(1)
                    else:
                        containment.append(0)

                contained.append(sum(containment) / (len(containment)))
                nonzero = 1 if sum(containment) > 0 else 0
                all_nonzero.append(nonzero)

        mean_containment = sum(contained) / len(contained) if contained else 0
        nonzero_fraction = sum(all_nonzero) / len(all_nonzero) if all_nonzero else 0

        metrics[prompt_type][key] = {
            "mean_containment": mean_containment,
            "nonzero_fraction": nonzero_fraction,
            "total_records": len(contained),
        }

# Create color scheme
colors = {"lora": "#2E86AB", "orig": "#A23B72", "diff": "#F18F01"}

# Figure 1: Mean Ground Truth Containment
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle("Mean Ground Truth Containment by Prompt Type", fontsize=16, fontweight="bold")
axes1 = axes1.flatten()

for idx, prompt_type in enumerate(PROMPT_TYPES):
    ax = axes1[idx]

    values = [metrics[prompt_type][key]["mean_containment"] for key in act_keys]
    x_pos = np.arange(len(act_keys))
    bars = ax.bar(x_pos, values, color=[colors[key] for key in act_keys], alpha=0.8, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(act_keys, fontsize=11)
    ax.set_ylabel("Mean Containment", fontsize=11)
    ax.set_title(prompt_type.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("mean_containment_by_prompt_type.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 2: Fraction of Non-Zero Records
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle("Fraction of Non-Zero Containment Records by Prompt Type", fontsize=16, fontweight="bold")
axes2 = axes2.flatten()

for idx, prompt_type in enumerate(PROMPT_TYPES):
    ax = axes2[idx]

    values = [metrics[prompt_type][key]["nonzero_fraction"] for key in act_keys]
    x_pos = np.arange(len(act_keys))
    bars = ax.bar(x_pos, values, color=[colors[key] for key in act_keys], alpha=0.8, edgecolor="black", linewidth=1.2)

    # Add value labels on bars (as percentages)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(act_keys, fontsize=11)
    ax.set_ylabel("Fraction Non-Zero", fontsize=11)
    ax.set_title(prompt_type.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("nonzero_fraction_by_prompt_type.png", dpi=300, bbox_inches="tight")
plt.show()

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
for prompt_type in PROMPT_TYPES:
    print(f"\n{prompt_type.upper().replace('_', ' ')}")
    print("-" * 70)
    for key in act_keys:
        m = metrics[prompt_type][key]
        print(
            f"  {key:6s} | Mean: {m['mean_containment']:.4f} | "
            f"Non-zero: {m['nonzero_fraction'] * 100:5.1f}% | "
            f"Records: {m['total_records']}"
        )
# %%
