#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
verl GRPO Training Launcher Script
Migrated from unsloth + trl to verl for GSM8K/MATH training

This script creates the configuration files and launches verl training.
Based on the verl documentation and examples.
"""

import os

from transformers import AutoTokenizer, PreTrainedTokenizer

from detection_eval.caller import read_jsonl_file_into_basemodel
from detection_eval.detection_basemodels import SAE
from detection_eval.steering_hooks import X_PROMPT, make_sae_verl_typed_dict

# set HF_HOME to /workspace
os.environ["HF_HOME"] = "/workspace"
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import wandb

# Step 2: Push to HuggingFace Hub
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel


class VerlParams(BaseModel):
    # Dataset paths
    train_path: str
    enable_gradient_checkpointing: bool = True
    eval_path: str | None = None
    reward_function_name: str = "compute_score"
    reward_function_file: str = "math_reward_function.py"
    experiment_name: str | None = None
    use_feature_vector: bool = True
    # Model configuration
    model_name: str = "google/gemma-2-9b-it"

    # Training configuration
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    gpu_memory_utilization: float = 0.6
    micro_batch: int = 16
    gradient_accumulation_steps: int = 1
    micro_batch_size_per_gpu: int = 8  # New parameter for fine control
    max_steps: int = 100
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4
    warmup_steps: int = 10
    beta: float = 0.005  # KL coefficient

    # SAE feature-vector config
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 131
    use_decoder_vectors: bool

    # LoRA configuration
    lora_rank: int = 32  # LoRA rank, set to 0 to disable LoRA
    lora_alpha: float = 64.0  # LoRA alpha parameter (typically 2x lora_rank)
    target_modules: str = "all-linear"  # Target modules for LoRA adaptation
    use_shm: bool = True  # Preload model into /dev/shm for faster loading
    layered_summon: bool = True  # Reduce GPU memory usage for large models

    # Output configuration
    output_dir: str = "./outputs"
    save_steps: int = 500
    log_steps: int = 1

    # HuggingFace Hub configuration
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hf_api_key: str | None = None

    # System configuration
    n_gpus: int = 1
    use_wandb: bool = True
    wandb_project: str = "gsm8k-verl-grpo"
    wandb_api_key: str | None = None


def get_sae_info(sae_repo_id: str, sae_width: int, sae_layer: int) -> str:
    if sae_repo_id == "google/gemma-scope-9b-it-res":
        if sae_width == 16:
            return f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            return f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")


def load_sae_params_for_model(
    sae_layer: int,
    sae_width: int,
    sae_repo_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download and load SAE params (W_enc, W_dec) for the tokenizer's model family."""
    filename = get_sae_info(sae_repo_id, sae_width, sae_layer)
    path_to_params = hf_hub_download(
        repo_id=sae_repo_id,
        filename=filename,
        force_download=False,
        local_dir="downloaded_saes",
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        torch.save(pt_params, pytorch_path)
    pt_params = torch.load(pytorch_path)
    W_enc = pt_params["W_enc"]  # [d_in, d_sae]
    W_dec = pt_params["W_dec"]  # [d_sae, d_in]
    return W_enc, W_dec


def get_feature_vector_from_params(
    W_enc: torch.Tensor,
    W_dec: torch.Tensor,
    sae_id: int,
    use_decoder_vectors: bool,
) -> list[float]:
    if use_decoder_vectors:
        return W_dec[sae_id].to(torch.float32).tolist()
    else:
        return W_enc[:, sae_id].to(torch.float32).tolist()


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags"""
    if "<answer>" in text and "</answer>" in text:
        after_ans = text.split("<answer>")[-1]
        answer = after_ans.split("</answer>")[0]
        return answer.strip()
    return text.strip()


def load_and_convert_dataset(
    model: str,
    tokenizer: PreTrainedTokenizer,
    dataset_path: str,
    output_path: str,
    sae_layer: int,
    sae_width: int,
    use_decoder_vectors: bool,
    sae_repo_id: str,
) -> int:
    """
    Load dataset from JSONL and convert to verl format (parquet).

    Args:
        dataset_path: Path to input JSONL file
        output_path: Path to output parquet file
        data_source: Source identifier for the dataset

    Returns:
        Number of samples processed


    class SAE(BaseModel):
        sae_id: int
        activations: SAEActivations
        # Sentences that do not activate for the given sae_id. But come from a similar SAE
        # Here the sae_id correspond to different similar SAEs.
        # The activations are the activations w.r.t this SAE. And should be low.
        hard_negatives: list[SAEActivations]

    """
    # Each line in jsonl should be SAE object

    print(f"Loading dataset from: {dataset_path}")

    # ---------------- build prompt and locate X position ----------------
    prompt_as_chat_dict = {
        "role": "user",
        "content": X_PROMPT,
    }
    tokenized_prompt = tokenizer.apply_chat_template(
        [prompt_as_chat_dict],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    # find positional index of the 'X' token within the prompt token ids
    try:
        position_idx = next(i for i, tid in enumerate(tokenized_prompt) if tid == x_token_id)
    except StopIteration:
        raise ValueError("Could not find token 'X' in the tokenized prompt")
    print(f"X token id: {x_token_id}; position index in prompt: {position_idx}")

    # ---------------- feature-vector params (loaded once) ----------------
    W_enc, W_dec = load_sae_params_for_model(
        sae_layer=sae_layer,
        sae_width=sae_width,
        sae_repo_id=sae_repo_id,
    )

    # ---------------- build rows ----------------
    testing_hack = False
    if model == "google/gemma-2-2b-it":
        print("WARNING: WE DON'T HAVE SAES FOR 2B, WILL USE 9B SAES")
        testing_hack = True

    data = []
    jsonl_items = read_jsonl_file_into_basemodel(dataset_path, SAE)
    sae_ids: list[int] = jsonl_items.map(lambda x: x.sae_id)
    if use_decoder_vectors:
        feature_vector_list: list[list[float]] = W_dec[sae_ids].tolist()
    else:
        # uhhh not sure if this is correct? but using decoder vectors for now.
        feature_vector_list: list[list[float]] = W_enc[:, sae_ids].T.tolist()

    for idx, sae in enumerate(jsonl_items):
        sample_dict = sae.model_dump()
        # Load the SAE train info. Should conform to SAE basemodel.
        sample = SAE.model_validate(sample_dict)
        feature = feature_vector_list[idx]

        if testing_hack:
            # ndim 2304 for 2b
            feature = feature[:2304]

        sae_verl_data = make_sae_verl_typed_dict(
            sample,
            position_idx,
            feature,
        )
        del sae_verl_data["activations"]  # TODO: add back in
        del sae_verl_data["hard_negatives"]  # TODO: add back in

        # Create structured data following the pattern
        structured_data = {
            "data_source": "custom",
            "prompt": [prompt_as_chat_dict],
            "ability": "explanations",
            "reward_model": {
                "style": "rule",
                "ground_truth": "no ground truth",
            },
            "extra_info": {
                "prompt": X_PROMPT,
                "index": idx,
            },
            # sae information which we modify the PPO trainer to pass during rollouts
            "sae": sae_verl_data,
        }
        data.append(structured_data)
        if idx == 0:
            print(f"First sample:\n {structured_data}")

    # Save as parquet for verl using pyarrow (no pandas)
    table = pa.Table.from_pylist(data)
    pq.write_table(table, output_path)

    print(f"Converted {len(data)} samples to {output_path}")
    return len(data)


def convert_verl_to_hf_and_push(params: VerlParams, step: int | None = None):
    """
    Convert verl checkpoint to HuggingFace format using verl model merger and push to Hub.

    Args:
        params: Training parameters with Hub configuration
        step: Training step number (for checkpoint naming)
    """
    if not params.push_to_hub or not params.hub_repo_id or not params.hf_api_key:
        print("Skipping HuggingFace push - not configured")
        return

    # Construct checkpoint paths based on verl's default structure
    project_name = params.wandb_project
    experiment_name = params.experiment_name or f"grpo-{params.model_name.split('/')[-1]}-no-beta"

    # Find the latest checkpoint if step not specified
    if step is None:
        # Look directly in output_dir for global_step directories
        if os.path.exists(params.output_dir):
            # Find the highest global_step directory
            step_dirs = [d for d in os.listdir(params.output_dir) if d.startswith("global_step_")]
            if step_dirs:
                step = max([int(d.split("_")[-1]) for d in step_dirs])
                print(f"Found latest checkpoint at step {step}")
            else:
                print("❌ No checkpoints found!")
                raise ValueError("No checkpoints found!")
        else:
            print(f"❌ Output directory not found: {params.output_dir}")
            raise ValueError(f"Output directory not found: {params.output_dir}")

    # Construct the checkpoint paths - simplified structure
    actor_checkpoint_dir = os.path.join(params.output_dir, f"global_step_{step}", "actor")
    hf_output_dir = os.path.join(params.output_dir, f"global_step_{step}", "actor", "huggingface")

    if not os.path.exists(actor_checkpoint_dir):
        print(f"❌ Actor checkpoint not found: {actor_checkpoint_dir}")
        raise ValueError(f"Actor checkpoint not found: {actor_checkpoint_dir}")

    # Check if this is a LoRA adapter checkpoint
    lora_adapter_dir = os.path.join(actor_checkpoint_dir, "lora_adapter")

    if os.path.exists(lora_adapter_dir):
        print("Found LoRA adapter - uploading LoRA adapter directly...")
        print(f"LoRA adapter source: {lora_adapter_dir}")

        # For LoRA adapters, use the lora_adapter directory directly
        hf_output_dir = lora_adapter_dir
        print("✅ Using LoRA adapter directory directly for upload")

    else:
        print("Converting full model checkpoint to HuggingFace format...")
        print(f"Source: {actor_checkpoint_dir}")
        print(f"Target: {hf_output_dir}")

        # Step 1: Convert verl checkpoint to HuggingFace format using verl model merger
        merge_cmd = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            actor_checkpoint_dir,
            "--target_dir",
            hf_output_dir,
        ]

        print(f"Running model merger: {' '.join(merge_cmd)}")
        subprocess.run(merge_cmd, capture_output=True, text=True, check=True)
        print("✅ Successfully converted checkpoint to HuggingFace format")

    # Determine repository name
    repo_name = f"{params.hub_repo_id}-step-{step}" if step else params.hub_repo_id

    print(f"Pushing to HuggingFace Hub: {repo_name}")

    # Create repo and upload the converted model
    api = HfApi()
    api.create_repo(repo_name, token=params.hf_api_key, exist_ok=True, private=False)

    # Upload the entire model directory
    api.upload_folder(
        folder_path=hf_output_dir,
        repo_id=repo_name,
        token=params.hf_api_key,
        commit_message=f"verl GRPO trained model at step {step}",
        ignore_patterns=["*.bin"],  # Upload safetensors instead of bin files
    )

    print(f"✅ Successfully pushed model to https://huggingface.co/{repo_name}")
    print("🎉 Model conversion and upload complete!")


def launch_verl_training(params: VerlParams, train_parquet: str, eval_parquet: str | None, reward_file: str):
    """
    Launch verl training by passing parameters directly to the subprocess.

    Args:
        params: Training parameters
        train_parquet: Path to training parquet file
        eval_parquet: Path to eval parquet file (optional)
        reward_file: Path to reward function file
    """

    # Validate LoRA configuration
    if params.lora_rank > 0:
        if params.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when LoRA is enabled")
        if not params.target_modules:
            raise ValueError("target_modules must be specified when LoRA is enabled")
        print(f"LoRA enabled: rank={params.lora_rank}, alpha={params.lora_alpha}, targets={params.target_modules}")
        print("Note: Using safetensors load format and increased learning rate for LoRA")
    else:
        print("LoRA disabled - using full parameter training")

    # Construct the verl training command with Hydra overrides
    max_num_batched_tokens = params.max_prompt_length + params.max_response_length

    # Set load format based on LoRA configuration
    if params.lora_rank > 0:
        load_format = "safetensors"
    else:
        load_format = "dummy_dtensor"

    bs = params.micro_batch * params.gradient_accumulation_steps

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "trainer.log_val_generations=10",
        # Data configuration
        f"data.train_files={train_parquet}",
        "data.prompt_key=prompt",
        f"data.max_prompt_length={params.max_prompt_length}",
        f"data.max_response_length={params.max_response_length}",
        f"data.train_batch_size={bs}",
        "data.shuffle=true",
        "data.truncation=error",
        "data.filter_overlong_prompts=true",
        # Algorithm configuration
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.kl_ctrl.type=fixed",
        f"algorithm.kl_ctrl.kl_coef={params.beta}",
        # Model configuration
        "actor_rollout_ref.hybrid_engine=true",
        f"actor_rollout_ref.model.path={params.model_name}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={params.enable_gradient_checkpointing}",
        "actor_rollout_ref.model.trust_remote_code=false",
        "actor_rollout_ref.model.use_remove_padding=true",
    ]

    if eval_parquet:
        cmd.append(f"data.val_files={eval_parquet}")
        cmd.append(f"trainer.val_batch_size={bs}")
    else:
        # need to still pass val_files. set a very high number for eval steps
        cmd.append(f"data.val_files={train_parquet}")
        cmd.append("trainer.test_freq=-1")

    if params.use_feature_vector:
        # use FeatureVectorRolloutRefWorker if we want to use feature vector steering.
        # Note: don't override actor_rollout_ref.rollout.mode because verl's code depends on it being set to sync in a few places.
        # I made a new key in ppo_trainer.yaml to allow this.
        # I HATE HYDRA AND YAML.
        cmd.append("actor_rollout_ref.use_feature_vector_steering=true")

    # Add LoRA parameters conditionally
    if params.lora_rank > 0:
        cmd.extend(
            [
                f"actor_rollout_ref.model.lora_rank={params.lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={params.lora_alpha}",
                f"actor_rollout_ref.model.target_modules={params.target_modules}",
                f"actor_rollout_ref.model.use_shm={str(params.use_shm).lower()}",
                f"actor_rollout_ref.rollout.layered_summon={str(params.layered_summon).lower()}",
            ]
        )

    # Continue with actor configuration
    cmd.extend(
        [
            # Actor configuration
            "actor_rollout_ref.actor.strategy=fsdp2",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={params.micro_batch}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
            "actor_rollout_ref.actor.ppo_epochs=1",
            "actor_rollout_ref.actor.grad_clip=0.5",
            "actor_rollout_ref.actor.clip_ratio=0.2",
            "actor_rollout_ref.actor.entropy_coeff=0.0",
            "actor_rollout_ref.actor.use_kl_loss=true",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            f"actor_rollout_ref.actor.optim.lr={params.learning_rate}",
            f"actor_rollout_ref.actor.optim.lr_warmup_steps={params.warmup_steps}",
            "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
            f"actor_rollout_ref.actor.optim.total_training_steps={params.max_steps}",
            "actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0",
            "actor_rollout_ref.actor.fsdp_config.param_offload=true",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
            # Reference model configuration
            "actor_rollout_ref.ref.strategy=fsdp2",
            "actor_rollout_ref.ref.fsdp_config.param_offload=true",
            "actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
            # Rollout configuration
            "actor_rollout_ref.model.use_fused_kernels=true",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.top_k=-1",
            "actor_rollout_ref.rollout.top_p=1.0",
            f"actor_rollout_ref.rollout.prompt_length={params.max_prompt_length}",
            f"actor_rollout_ref.rollout.response_length={params.max_response_length}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
            "actor_rollout_ref.rollout.dtype=bfloat16",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={params.gpu_memory_utilization}",
            "actor_rollout_ref.rollout.ignore_eos=false",
            "actor_rollout_ref.rollout.enforce_eager=true",
            "actor_rollout_ref.rollout.free_cache_engine=true",
            f"actor_rollout_ref.rollout.load_format={load_format}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            f"actor_rollout_ref.rollout.n={params.num_generations}",
            "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
            "reward_model.reward_manager=batch",
            # Reward model configuration
            "reward_model.enable=false",
            # Custom reward function
            f"custom_reward_function.path={reward_file}",
            f"custom_reward_function.name={params.reward_function_name}",
            # Trainer configuration
            "trainer.total_epochs=1",
            f"trainer.project_name={params.wandb_project}",
            f"trainer.experiment_name=grpo-{params.model_name.split('/')[-1]}",
            "trainer.nnodes=1",
            f"trainer.n_gpus_per_node={params.n_gpus}",
            f"trainer.save_freq={params.save_steps}",
            f"trainer.test_freq={params.save_steps}",
            "trainer.val_before_train=false",
            f"trainer.default_local_dir={params.output_dir}",
        ]
    )

    # Add logger configuration
    if params.use_wandb:
        cmd.append('trainer.logger=["console","wandb"]')
    else:
        cmd.append("trainer.logger=console")

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(params.n_gpus))
    # env["RAY_DEBUG_POST_MORTEM"] = "1" # I couldn't get post mortem to work properly

    if params.use_wandb:
        wandb_key = params.wandb_api_key
        assert wandb_key, "WANDB_API_KEY is required for wandb logging"

    print("Launching verl training with direct parameters...")
    print(f"Using GPUs: {env.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Command: {' '.join(cmd[:6])} ... [+{len(cmd) - 6} more args]")

    # Launch training
    subprocess.run(cmd, env=env, check=True)
    print("Training completed successfully!")

    # After training completes, convert and push final model to HuggingFace
    if params.push_to_hub:
        print("\nConverting final model to HuggingFace format...")
        convert_verl_to_hf_and_push(params)


def verl_main(params: VerlParams):
    """Main training pipeline using verl"""

    # Validate required environment variables (like your current script)
    if params.push_to_hub:
        if not params.hf_api_key:
            print("❌ Error: HF_WRITE_TOKEN environment variable is required for HuggingFace push")
            sys.exit(1)
        if not params.hub_repo_id:
            print("❌ Error: hub_repo_id must be provided when push_to_hub=True")
            sys.exit(1)

    if params.use_wandb:
        assert params.wandb_api_key, "WANDB_API_KEY is required for wandb logging"
        wandb.login(key=params.wandb_api_key)

    print("Starting verl GRPO training setup...")
    print(f"Model: {params.model_name}")
    print(f"Train data: {params.train_path}")
    print(f"Eval data: {params.eval_path}")
    print(f"Output dir: {params.output_dir}")
    if params.push_to_hub:
        print(f"Will push to HuggingFace Hub: {params.hub_repo_id}")

    # Create output directory
    os.makedirs(params.output_dir, exist_ok=True)

    # Load tokenizer once and pass down
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    # Convert datasets to parquet format
    train_parquet = os.path.join(params.output_dir, "train.parquet")
    load_and_convert_dataset(
        params.model_name,
        tokenizer,
        params.train_path,
        train_parquet,
        sae_layer=params.sae_layer,
        sae_width=params.sae_width,
        use_decoder_vectors=params.use_decoder_vectors,
        sae_repo_id=params.sae_repo_id,
    )

    eval_parquet = None
    if params.eval_path:
        eval_parquet = os.path.join(params.output_dir, "eval.parquet")
        load_and_convert_dataset(
            params.model_name,
            tokenizer,
            params.eval_path,
            eval_parquet,
            sae_layer=params.sae_layer,
            sae_width=params.sae_width,
            use_decoder_vectors=params.use_decoder_vectors,
            sae_repo_id=params.sae_repo_id,
        )

    # Use math reward function directly
    reward_file = params.reward_function_file
    assert os.path.exists(reward_file), f"Reward function file not found: {reward_file}"

    # Launch training
    print("\nLaunching verl training...")
    launch_verl_training(params, train_parquet, eval_parquet, reward_file)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Load environment variables
    hf_api_key = os.getenv("HF_WRITE_TOKEN")
    wandb_key = os.getenv("WANDB_KEY")

    # Configuration (optimized based on reference GRPO setup)
    params = VerlParams(
        # smaller model for testing
        # model_name="google/gemma-2-2b-it",
        model_name="thejaminator/gemma-introspection-20250821-merged",  # loras don't get merged automatically
        # sae_repo_id="google/gemma-scope-9b-it-res",
        use_feature_vector=True,  # debugging logprobs
        train_path="hard_negatives_100_000_to_100_800.jsonl",
        max_seq_length=1_000,  # debug
        max_prompt_length=500,  # debug
        max_response_length=2_000,  # debug
        num_generations=4,  # Bigger group size since noisy explanations
        gpu_memory_utilization=0.4,  # some other thing running
        # model_name="google/gemma-2-9b-it",
        # num_generations=16,  # Bigger group size since noisy explanations
        # max_seq_length=8_000,  # More reasonable for math problems
        # max_prompt_length=2_000,  # Reduced from 6000, matching reference
        # max_response_length=6_000,  # Reduced from 6000, matching reference
        # micro_batch=8,
        # micro_batch_size_per_gpu=8,
        micro_batch=2,  # for test purposes, usually 4
        micro_batch_size_per_gpu=2,  # for test purposes, usually 4
        warmup_steps=5,
        gradient_accumulation_steps=1,  # for test purposes, usually 4
        learning_rate=5e-5,  # Increased by order of magnitude for LoRA (was 5e-6)
        beta=0.01,  # KL penalty
        # LoRA configuration for 4B model (following best practices)
        lora_rank=64,  # Recommended >=32 for good convergence, using 64 for 4B model
        lora_alpha=128.0,  # Typically 2x lora_rank
        target_modules="all-linear",  # Apply LoRA to all linear layers
        use_shm=False,
        layered_summon=False,
        max_steps=4000,
        output_dir="/workspace/verl_outputs_feature_vector",
        eval_path=None,
        save_steps=10,
        n_gpus=1,
        use_wandb=True,
        wandb_project="grpo-feature-vector",
        # HuggingFace Hub configuration (like your current script)
        push_to_hub=True,
        hub_repo_id="thejaminator/grpo-feature-vector",  # Updated with "_verl" suffix
        hf_api_key=hf_api_key,
        reward_function_name="compute_score",
        reward_function_file="feature_vector_reward.py",
        wandb_api_key=wandb_key,
        use_decoder_vectors=True,
        sae_layer=9,
        sae_width=131,
        enable_gradient_checkpointing=False,
    )

    verl_main(params)
