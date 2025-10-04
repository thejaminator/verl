"""
Standalone LatentQA dataset loader (no tokenization, no collator, no templates).

This file consolidates the logic needed to load the LatentQA dataset from the
JSON files (system, stimulus_completion, stimulus, control, qa) and produce a
Python dataset object whose items contain:

- read_prompt: list of {"role": str, "content": str} messages to feed your model
- dialog:      list of two messages: user question and assistant answer
- mask_type:   either "system" or "user" (useful if you later decide to mask)
- label:       the data label from the JSON
- source:      which split produced the item: one of
               ["system", "stimulus_completion", "stimulus", "control"]

Notes and simplifications:
- No tokenizer, no special chat templates, no magic token offsets.
- Returns raw message dicts (role/content). You can format them however you like.
- Balances are assumed across labels (asserts like the original code).
- Optional filtering by label prefixes and optional subsampling (train_percent).

Example usage:

    from latentqa_dataset_standalone import (
        DataPaths, load_latentqa_dataset, preview_dataset
    )

    paths = DataPaths(
        system="data/train/system.json",
        stimulus_completion="data/train/stimulus_completion.json",
        stimulus="data/train/stimulus.json",
        control="data/train/control.json",
        qa="data/train/qa.json",
    )

    dataset = load_latentqa_dataset(
        paths,
        filter_prefixes=[],   # or e.g., ["goal"] to drop labels starting with "goal-"
        train_percent=1.0,
        add_thought_tokens=False,
        seed=42,
    )

    print(len(dataset))
    item = dataset[0]
    print(item["label"], item["source"], item["mask_type"])
    print(item["read_prompt"])   # list of {role, content}
    print(item["dialog"])        # [ {user}, {assistant} ]

    # Quick peek at a few examples for sanity:
    preview_dataset(dataset, limit=3)

"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


@dataclass
class DataPaths:
    """Container for dataset file paths.

    Any path can be an empty string or None to disable that component.
    """

    system: Optional[str] = None
    stimulus_completion: Optional[str] = None
    stimulus: Optional[str] = None
    control: Optional[str] = None
    qa: Optional[str] = None


# --------------------------------------------------------------------------------------
# JSON loading helpers
# --------------------------------------------------------------------------------------


def _read_json(path: Optional[str]) -> Any:
    if not path:
        return None
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_behavior_item(item: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    """Return a 7-tuple corresponding to the unified behavior format.

    Order matches the original repo logic:
    (system, control_user, control_thought, control_model, stimulus_user, stimulus_thought, stimulus_model)
    Missing fields default to empty strings.
    """

    return (
        item.get("system", ""),
        item.get("control_user", ""),
        item.get("control_thought", ""),
        item.get("control_model", ""),
        item.get("stimulus_user", ""),
        item.get("stimulus_thought", ""),
        item.get("stimulus_model", ""),
    )


def _build_data_and_id_tuples(
    path: Optional[str],
    qa_data: Dict[str, List[List[str]]],
    filter_prefixes: Sequence[str],
    train_percent: float,
    rng: random.Random,
) -> Tuple[Dict[str, List[Tuple[str, str, str, str, str, str, str]]], List[Tuple[int, int, int]]]:
    """Load one of the behavior JSON files and produce (data_by_label, id_tuples).

    - data_by_label: dict[label] -> list of 7-tuples behavior entries
    - id_tuples: list of (label_idx, data_idx, qa_idx), sampled per train_percent
    """

    data_by_label: Dict[str, List[Tuple[str, str, str, str, str, str, str]]] = {}
    if not path:
        return data_by_label, []

    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Behavior file must be a list of items: {path}")

    # Optional label filtering: drop items where the first label prefix matches any in filter_prefixes
    for item in raw:
        label: str = item.get("label", "")
        first_prefix = label.split("-")[0] if label else ""
        if filter_prefixes and first_prefix in filter_prefixes:
            continue
        data_by_label.setdefault(label, []).append(_normalize_behavior_item(item))

    if not data_by_label:
        return {}, []

    # All labels should have the same number of behaviors (assert like original code)
    lens = [len(v) for v in data_by_label.values()]
    num_behaviors_max = max(lens)
    num_behaviors_min = min(lens)
    if num_behaviors_max != num_behaviors_min:
        raise ValueError(
            "All labels must have the same number of behaviors per behavior file. "
            f"Found min={num_behaviors_min}, max={num_behaviors_max} in {path}"
        )
    num_behaviors = num_behaviors_max

    # All labels must have the same number of QAs
    qa_lens = [len(qa_data[label]) for label in data_by_label]
    num_qa_max = max(qa_lens)
    num_qa_min = min(qa_lens)
    if num_qa_max != num_qa_min:
        raise ValueError(
            f"All labels must have the same number of QA pairs. Found min={num_qa_min}, max={num_qa_max} in QA file"
        )
    num_qa = num_qa_max

    # Build id tuples over label x behavior x qa
    labels_list = list(data_by_label.keys())
    total = len(labels_list) * num_behaviors * num_qa
    all_ids = list(range(total))

    if train_percent < 1.0:
        keep = max(1, int(total * train_percent))
        all_ids = rng.sample(all_ids, keep)

    id_tuples: List[Tuple[int, int, int]] = []
    for x in all_ids:
        label_idx = x // (num_behaviors * num_qa)
        data_idx = (x // num_qa) % num_behaviors
        qa_idx = x % num_qa
        id_tuples.append((label_idx, data_idx, qa_idx))

    return data_by_label, id_tuples


# --------------------------------------------------------------------------------------
# Dataset object (no torch dependency required)
# --------------------------------------------------------------------------------------


class LatentQADatasetSimple:
    """Simple indexable dataset that emits raw chat messages and QA pairs.

    Each item is a dict with keys: label, source, read_prompt, dialog, mask_type.
    """

    SOURCES = ("system", "stimulus_completion", "stimulus", "control")

    def __init__(
        self,
        data_groups: Sequence[Dict[str, List[Tuple[str, str, str, str, str, str, str]]]],
        id_groups: Sequence[List[Tuple[int, int, int]]],
        qa_data: Dict[str, List[List[str]]],
        *,
        add_thought_tokens: bool = False,
    ) -> None:
        assert len(data_groups) == len(self.SOURCES)
        assert len(id_groups) == len(self.SOURCES)

        self.data_groups = data_groups
        self.id_groups = id_groups
        self.qa_data = qa_data
        self.add_thought_tokens = add_thought_tokens

        # Store labels list per group to enable index-to-label mapping
        self.labels_per_group: List[List[str]] = [list(d.keys()) for d in self.data_groups]

        # Build a flat index mapping for simplicity
        # Each entry is (group_idx, label_idx, data_idx, qa_idx)
        self.index: List[Tuple[int, int, int, int]] = []
        for g, tuples in enumerate(self.id_groups):
            for label_idx, data_idx, qa_idx in tuples:
                self.index.append((g, label_idx, data_idx, qa_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        g, label_idx, data_idx, qa_idx = self.index[idx]
        group_name = self.SOURCES[g]
        labels = self.labels_per_group[g]
        label = labels[label_idx]

        behavior = self.data_groups[g][label][data_idx]
        qa_pair = self.qa_data[label][qa_idx]

        (
            system,
            control_user,
            control_thought,
            control_model,
            stimulus_user,
            stimulus_thought,
            stimulus_model,
        ) = behavior

        # Build read_prompt and mask_type like the original dataset, but without templates
        if system != "":
            # System-only case
            read_prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": stimulus_user},
            ]
            mask_type = "system"
        elif control_model == "":
            # Control-only case
            read_prompt = [
                {"role": "user", "content": control_user},
            ]
            mask_type = "user"
        elif stimulus_model == "":
            # Control followed by user stimulus (no final model message)
            read_prompt = [
                {"role": "user", "content": control_user},
                {"role": "assistant", "content": control_model},
                {"role": "user", "content": stimulus_user},
            ]
            mask_type = "user"
        else:
            # Full conversation including final stimulus model answer
            if self.add_thought_tokens:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {
                        "role": "assistant",
                        "content": f"<think>\n{stimulus_thought}\n</think>\n\n{stimulus_model}",
                    },
                ]
            else:
                read_prompt = [
                    {"role": "user", "content": control_user},
                    {"role": "assistant", "content": control_model},
                    {"role": "user", "content": stimulus_user},
                    {"role": "assistant", "content": stimulus_model},
                ]
            mask_type = "user"

        qa_dialog = [
            {"role": "user", "content": qa_pair[0]},
            {"role": "assistant", "content": qa_pair[1]},
        ]

        return {
            "label": label,
            "source": group_name,
            "read_prompt": read_prompt,
            "dialog": qa_dialog,
            "mask_type": mask_type,
        }


# --------------------------------------------------------------------------------------
# Public loader API
# --------------------------------------------------------------------------------------


def load_latentqa_dataset(
    paths: DataPaths,
    *,
    filter_prefixes: Sequence[str] | None = None,
    train_percent: float = 1.0,
    add_thought_tokens: bool = False,
    seed: int = 42,
) -> LatentQADatasetSimple:
    """Load all dataset components and return a LatentQADatasetSimple.

    Arguments
    - paths: DataPaths with JSON file paths.
    - filter_prefixes: optional list of label prefixes to exclude (e.g., ["goal"]).
    - train_percent: 0 < fraction <= 1.0 to randomly subsample id tuples per source.
    - add_thought_tokens: if True, final assistant turn includes <think>...</think> content.
    - seed: RNG seed for subsampling when train_percent < 1.
    """

    if not paths.qa:
        raise ValueError("QA file path (paths.qa) is required")

    filter_prefixes = list(filter_prefixes or [])
    rng = random.Random(seed)

    qa_data = _read_json(paths.qa)
    if not isinstance(qa_data, dict):
        raise ValueError("QA file must be a dict of label -> list[[question, answer], ...]")

    # Build groups in the canonical order used elsewhere
    data_groups: List[Dict[str, List[Tuple[str, str, str, str, str, str, str]]]] = []
    id_groups: List[List[Tuple[int, int, int]]] = []

    for path in (paths.system, paths.stimulus_completion, paths.stimulus, paths.control):
        data_by_label, id_tuples = _build_data_and_id_tuples(path, qa_data, filter_prefixes, train_percent, rng)
        data_groups.append(data_by_label)
        id_groups.append(id_tuples)

    return LatentQADatasetSimple(
        data_groups=data_groups,
        id_groups=id_groups,
        qa_data=qa_data,
        add_thought_tokens=add_thought_tokens,
    )


# --------------------------------------------------------------------------------------
# Small utilities for quick checks
# --------------------------------------------------------------------------------------


def preview_dataset(
    dataset: LatentQADatasetSimple,
    *,
    per_source: int = 2,
) -> List[Dict[str, Any]]:
    """Print and return up to `per_source` items for each present source.

    Returns a list of compact dict summaries in the printed order.
    """
    want = {s: per_source for s in dataset.SOURCES}
    out: List[Dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        src = sample["source"]
        if want.get(src, 0) <= 0:
            continue
        roles = [turn["role"] for turn in sample["dialog"]]
        read_roles = [turn["role"] for turn in sample["read_prompt"]]
        read_excerpt = sample["read_prompt"][0]["content"][:120] if sample["read_prompt"] else ""
        print(f"\n\n\n=== IDX {i} | label={sample['label']} | source={src} | mask={sample['mask_type']} ===")
        print(f"read_prompt[0][:120]: {read_excerpt}")
        print(f"dialog roles: {roles}")
        print(f"read_roles: {read_roles}")
        print(f"\nQ/A: {sample['dialog']}")
        print(f"\nread_prompt: {sample['read_prompt']}")
        out.append(
            {
                "index": i,
                "label": sample["label"],
                "source": src,
                "mask_type": sample["mask_type"],
                "dialog_roles": roles,
                "read_prompt_excerpt": read_excerpt,
            }
        )
        want[src] -= 1
        if all(v <= 0 for v in want.values()):
            break
    for s, remaining in want.items():
        if remaining == per_source:
            print(f"\n(no items for source: {s})")
    return out
