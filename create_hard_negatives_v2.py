#!/usr/bin/env python3
"""
Script to find similar SAE features and verify hard negatives using the Gemma 9B model.

This script:
1. Loads a SAE and finds the most similar features to a target feature
2. Loads the Gemma 9B model and computes actual SAE activations on sentences
3. Identifies sentences from similar features that don't activate for the target feature (hard negatives)
4. Outputs results to JSONL format

Usage as a module:
    from compare_and_verify_hard_negatives import main
    main(target_features=[0, 1, 2], num_sentences=5, top_k_similar_features=10, batch_size=20)

Or modify the call at the bottom of this file and run directly:
    python compare_and_verify_hard_negatives.py

TODO:
1. If activation is 0, don't write the key
2. no need to dump the full sentence
3. as_str -> str?
4. no need token id
5. round activaiton to 2 d.p
"""

import json
import os
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import einops
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from tqdm import tqdm

from detection_eval.detection_basemodels import (
    SAEV2,
    SAEActivationsV2,
    SentenceInfoV2,
    TokenActivationV2,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

# Copied from interp_tools to make file standalone
import contextlib
from abc import ABC, abstractmethod

import numpy as np
from huggingface_hub import hf_hub_download


class SAEInfo(NamedTuple):
    sae_width: int
    sae_layer: int
    sae_layer_percent: int
    sae_filename: str


def get_sae_info(sae_repo_id: str, sae_layer_percent: int = 25, sae_width: int | None = None) -> SAEInfo:
    if sae_repo_id == "google/gemma-scope-9b-it-res":
        num_layers = 42
        assert sae_layer_percent == 25
        sae_layer = 9

        # Gemma scope IT saes: https://huggingface.co/google/gemma-scope-9b-it-res/tree/main
        assert sae_layer in [9, 20, 31]

        # Note: For gemma_scope saes you need to specify the L0 if you use different layers / widths

        if sae_width is None:
            sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        num_layers = 32

        assert sae_layer_percent == 25
        sae_layer = int(num_layers * (sae_layer_percent / 100))

        assert sae_layer in [8, 16, 24]

        if sae_width is None:
            sae_width = 32
        sae_filename = ""
    elif sae_repo_id == "adamkarvonen/qwen3-8b-saes":
        num_layers = 36
        sae_layer = int(num_layers * (sae_layer_percent / 100))

        # Only have these SAEs available: https://huggingface.co/adamkarvonen/qwen3-8b-saes/tree/main
        assert sae_layer in [9, 18, 27]

        if sae_width is None:
            sae_width = 2
        sae_filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{sae_layer}/trainer_{sae_width}/ae.pt"
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return SAEInfo(
        sae_width=sae_width,
        sae_layer=sae_layer,
        sae_layer_percent=sae_layer_percent,
        sae_filename=sae_filename,
    )


# Configuration variables - no longer need a config class


# SAE Classes
class BaseSAE(torch.nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer
        self.d_sae = d_sae

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = 1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False


class JumpReluSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.threshold = torch.nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.d_sae = d_sae
        self.d_in = d_in

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Doing this because npz files are often insanely slow to load
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}

        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReluSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder norms are not normalized. Implement a normalization method.")

    return sae


class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=dtype, device=device))

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = torch.nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if self.use_threshold:
            if self.threshold < 0:  # type: ignore
                raise ValueError("Threshold is not set. The threshold must be set to use it during inference")
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)  # type: ignore
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)  # type: ignore

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_batch_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> BatchTopKSAE:
    assert "ae.pt" in filename, f"Filename {filename} does not contain 'ae.pt'"

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    # assert model_name in config["trainer"]["lm_name"], f"Model name {model_name} not in config {config['trainer']['lm_name']}"

    k = config["trainer"]["k"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
        "threshold": "threshold",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = BatchTopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


def load_sae(
    sae_repo_id: str,
    sae_filename: str,
    sae_layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseSAE:
    print(f"Loading SAE for layer {sae_layer} from {sae_repo_id}...")

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae = load_gemma_scope_jumprelu_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            layer=sae_layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    elif sae_repo_id == "adamkarvonen/qwen3-8b-saes":
        sae = load_dictionary_learning_batch_topk_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            layer=sae_layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")

    return sae


# Model utilities
def get_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path  # type: ignore

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]  # type: ignore
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]  # type: ignore
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]  # type: ignore
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)  # type: ignore
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD  # type: ignore


# Pydantic schema classes for JSONL output
def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """Load the max activating examples data."""
    acts_dir = "max_acts"

    if "gemma" in model_name:
        # Construct filename
        acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
            "/", "_"
        )

        acts_path = os.path.join(acts_dir, acts_filename)

    elif "Qwen" in model_name:
        acts_filename = f"acts_Qwen_Qwen3-8B_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt"
        acts_path = os.path.join(acts_dir, acts_filename)

    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_path}")
        try:
            hf_hub_download(
                repo_id="adamkarvonen/sae_max_acts",
                filename=acts_filename,
                force_download=False,
                local_dir=acts_dir,
                repo_type="dataset",
            )
            print(f"✅ Downloaded to: {acts_path}")
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            raise

    print(f"📂 Loading max acts data from: {acts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_data = torch.load(acts_path, map_location=device)

    return acts_data


@dataclass(kw_only=True)
class SimilarFeature:
    """Represents a feature similar to a target feature."""

    feature_idx: int
    similarity_score: float


def find_most_similar_features(
    sae, target_feature_idx: int, top_k: int = 1, exclude_self: bool = True
) -> list[SimilarFeature]:
    """Find the most similar features to a target feature using cosine similarity of encoder vectors."""
    # Get encoder weights - shape: [d_in, d_sae]
    W_enc = sae.W_enc.data

    # Get the target feature vector - shape: [d_in]
    target_vector = W_enc[:, target_feature_idx]

    # Compute cosine similarity with all other features
    # Normalize the target vector
    target_normalized = F.normalize(target_vector.unsqueeze(0), dim=1)

    # Normalize all encoder vectors
    all_vectors_normalized = F.normalize(W_enc.T, dim=1)  # Shape: [d_sae, d_in]

    # Compute cosine similarities - shape: [d_sae]
    similarities = torch.mm(all_vectors_normalized, target_normalized.T).squeeze()

    if exclude_self:
        # Set similarity to target feature itself to -inf so it's not selected
        similarities[target_feature_idx] = float("-inf")

    # Get top-k most similar features
    top_similarities, top_indices = torch.topk(similarities, k=top_k, largest=True)

    # Create SimilarFeature objects
    similar_features = []
    for sim_score, feature_idx in zip(top_similarities, top_indices, strict=False):
        similar_features.append(
            SimilarFeature(
                feature_idx=int(feature_idx.item()),
                similarity_score=float(sim_score.item()),
            )
        )

    return similar_features


def load_model_and_sae(
    model_name: str,
    sae_repo_id: str,
    sae_filename: str,
    sae_layer: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, object, torch.nn.Module]:
    """Load the Gemma 9B model, tokenizer, SAE, and submodule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    print("🧠 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load SAE
    print("🔧 Loading SAE...")
    sae = load_sae(sae_repo_id, sae_filename, sae_layer, model_name, device, dtype)

    # Get submodule for activation collection
    submodule = get_submodule(model, sae_layer)  # type: ignore

    return model, tokenizer, sae, submodule  # type: ignore


def compute_sae_activations_for_sentences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: object,
    submodule: torch.nn.Module,
    tokens_BL: torch.Tensor,
    target_feature_idx: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Compute SAE activations for a list of sentences and return SentenceInfo objects.
    """

    all_acts_BL = []

    # Process sentences in batches
    for i in range(0, tokens_BL.shape[0], batch_size):
        batch_tokens_BL = tokens_BL[i : i + batch_size]
        attn_mask_BL = torch.ones_like(batch_tokens_BL)

        tokenized = {
            "input_ids": batch_tokens_BL,
            "attention_mask": attn_mask_BL,
        }

        with torch.no_grad():
            # Get model activations at the SAE layer for the whole batch
            layer_acts_BLD = collect_activations(model, submodule, tokenized)

            # Encode through SAE
            encoded_acts_BLF = sae.encode(layer_acts_BLD)  # type: ignore

            norms_BL = torch.norm(layer_acts_BLD, dim=-1)
            median_norm = norms_BL.median()
            norm_mask_BL = norms_BL < median_norm * 10

            norm_mask_BL *= attn_mask_BL.bool()

            if tokenizer.bos_token_id is not None:
                bos_mask_BL = batch_tokens_BL != tokenizer.bos_token_id
                norm_mask_BL *= bos_mask_BL

            encoded_acts_BLF *= norm_mask_BL[:, :, None]

            encoded_acts_BL = encoded_acts_BLF[:, :, target_feature_idx]

            all_acts_BL.append(encoded_acts_BL)

    all_acts_BL = torch.cat(all_acts_BL, dim=0)

    return all_acts_BL


def main(
    target_features: Sequence[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    target_sentences: int = 20,
    top_k_similar_features: int = 10,
    negative_sentences: int = 8,  # we don't need so many
    output: str = "hard_negatives_results.jsonl",
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32,
    hard_negative_threshold: float = 0.05,
    batch_size: int = 20,
    sae_layer_percent: int = 25,
    verbose: bool = False,
):
    # check if output file exists
    if os.path.exists(output) and False:
        print(f"🔍 Output file {output} already exists. Not going to overwrite it.")
        return

    # Get SAE info
    sae_width, sae_layer, sae_layer_percent, sae_filename = get_sae_info(sae_repo_id, sae_layer_percent)

    print("🔧 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   SAE: {sae_repo_id}")
    print(f"   Layer: {sae_layer}")
    print(f"   Width: {sae_width}")
    print(f"   Target Features: {target_features}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Output: {output}")

    # Load max acts data
    print("📊 Loading max acts data...")
    acts_data = load_max_acts_data(
        model_name,
        sae_layer,
        sae_width,
        sae_layer_percent,
        context_length,
    )

    # Validate feature indices
    max_feature_idx = acts_data["max_tokens"].shape[0] - 1
    for feature_idx in target_features:
        if feature_idx > max_feature_idx:
            raise ValueError(f"Feature {feature_idx} not found. Max feature index: {max_feature_idx}")

    # Load model, tokenizer, and SAE
    print("🚀 Loading model and SAE...")
    model, tokenizer, sae, submodule = load_model_and_sae(model_name, sae_repo_id, sae_filename, sae_layer)
    # how many features in sae?
    print(f"🔍 Number of features in SAE: {len(sae.W_dec)}")  # type: ignore

    # Process each feature index

    # open file to append
    with open(output, "a") as f:
        for feature_idx in tqdm(target_features, desc="Processing features"):
            similar_features = find_most_similar_features(sae, feature_idx, top_k=top_k_similar_features)

            pos_tokens_BL = acts_data["max_tokens"][feature_idx, :target_sentences]

            pos_acts_BL = acts_data["max_acts"][feature_idx, :target_sentences]

            max_target_act = pos_acts_BL.max()

            all_similar_tokens_BL = []

            similar_feature_indices = [similar_feature.feature_idx for similar_feature in similar_features]

            all_similar_tokens_KBL = acts_data["max_tokens"][similar_feature_indices, :negative_sentences]

            all_similar_tokens_BL = einops.rearrange(all_similar_tokens_KBL, "K B L -> (K B) L")

            all_similar_acts_BL = compute_sae_activations_for_sentences(
                model,
                tokenizer,
                sae,
                submodule,
                all_similar_tokens_BL,
                feature_idx,
                batch_size,
            )
            if verbose:
                print(f"\n🎯 Processing feature {feature_idx}...")
                # Find most similar features
                print(f"🔍 Finding {top_k_similar_features} most similar features to feature {feature_idx}...")

                # Get sentences for target feature
                print(f"📝 Getting sentences for target feature {feature_idx}...")

                # Compute actual SAE activations for target feature sentences
                print("🧮 Computing SAE activations for target feature sentences...")

                # Collect all sentences from similar features first
                print(f"📝 Collecting sentences from {len(similar_features)} similar features...")

                # Compute target feature activations on ALL similar feature sentences in batches
                print(
                    f"🧮 Computing SAE activations for {all_similar_tokens_BL.shape[0]} sentences from similar features..."
                )

            max_similar_acts_B = all_similar_acts_BL.max(dim=1).values
            hard_negatives_mask_B = max_similar_acts_B < (hard_negative_threshold * max_target_act)
            hard_negatives_BL = all_similar_tokens_BL[hard_negatives_mask_B]

            if verbose:
                print(f"Found {hard_negatives_BL.shape[0]} hard negatives")

            decoded_hard_negatives = tokenizer.batch_decode(hard_negatives_BL, skip_special_tokens=True)
            decoded_pos_sentences = tokenizer.batch_decode(pos_tokens_BL, skip_special_tokens=True)

    # Write all results to JSONL
    print(f"\n💾 Writing results to {output}...")

    print("✅ Analysis complete!")
    print(f"   Features processed: {target_features}")
    print(f"   Results saved to: {output}")


if __name__ == "__main__":
    # Example usage - customize the feature_idxs and other parameters as needed
    # target_features = list(range(0, 100_000))
    # to_100k = list(range(0, 100_000))
    # 100k to 100_200
    # target_features = list(range(0, 200))
    min_idx = 0
    max_idx = 10_000
    # max_idx = 30
    target_features = list(range(min_idx, max_idx))

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    for sae_layer_percent in [25, 50, 75]:
        main(
            # model_name="google/gemma-2-9b-it",
            # sae_repo_id="google/gemma-scope-9b-it-res",
            model_name="Qwen/Qwen3-8B",
            sae_repo_id="adamkarvonen/qwen3-8b-saes",
            target_features=target_features,
            top_k_similar_features=34,
            batch_size=1024,
            target_sentences=32,
            output=f"{data_folder}/qwen_hard_negatives_{min_idx}_{max_idx}_layer_percent_{sae_layer_percent}.jsonl",
            sae_layer_percent=sae_layer_percent,
            verbose=False,
        )
