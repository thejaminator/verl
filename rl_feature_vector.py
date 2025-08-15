#!/usr/bin/env python3
"""
verl GRPO Training Launcher Script
Migrated from unsloth + trl to verl for GSM8K/MATH training

This script creates the configuration files and launches verl training.
Based on the verl documentation and examples.
"""

import json
import os

from create_hard_negative_and_feature_vector import SAE

# set HF_HOME to /workspace
os.environ["HF_HOME"] = "/workspace"
import subprocess
import sys
from typing import Sequence

import wandb

# Step 2: Push to HuggingFace Hub
from huggingface_hub import HfApi
from pydantic import BaseModel



class VerlParams(BaseModel):
    # Dataset paths
    train_path: str
    eval_path: str | None = None
    reward_function_name: str = "compute_score"
    reward_function_file: str = "math_reward_function.py"
    experiment_name: str | None = None
    actor_rollout_ref_strategy: str = "feature_vector"
    # Model configuration
    model_name: str = "google/gemma-2-9b-it"

    # Training configuration
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    micro_batch: int = 16
    gradient_accumulation_steps: int = 1
    micro_batch_size_per_gpu: int = 8  # New parameter for fine control
    max_steps: int = 100
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4
    warmup_steps: int = 10
    beta: float = 0.005  # KL coefficient

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


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags"""
    if "<answer>" in text and "</answer>" in text:
        after_ans = text.split("<answer>")[-1]
        answer = after_ans.split("</answer>")[0]
        return answer.strip()
    return text.strip()


def load_and_convert_dataset(dataset_path: str, output_path: str, data_source: str = "custom") -> int:
    """
    Load dataset from JSONL and convert to verl format (parquet).

    Args:
        dataset_path: Path to input JSONL file
        output_path: Path to output parquet file
        data_source: Source identifier for the dataset

    Returns:
        Number of samples processed
    """
    import pandas as pd
    # Each line in jsonl should be SAE object

    print(f"Loading dataset from: {dataset_path}")
    X_PROMPT = "Can you explain to me what 'X' means? Format your final answer with <explanation>"
    prompt_as_chat_dict = {
        "role": "user",
        "content": X_PROMPT,
    }

    data = []
    with open(dataset_path) as f:
        for idx, line in enumerate(f):
            if line.strip():
                sample_dict = json.loads(line)
                sample = SAE.model_validate(sample_dict)

                # Create structured data following the pattern
                structured_data = {
                    "data_source": data_source,
                    "prompt": [prompt_as_chat_dict],
                    "ability": "explanations",  # No idea what we need to edit this for, but verl requires it?
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": "no ground truth",
                    },  # verl requires passing something for ground truth?
                    "extra_info": {
                        "prompt": X_PROMPT,
                        "sae": sample.model_dump(),
                        "index": idx,
                    },
                }
                data.append(structured_data)

    # Save as parquet for verl
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

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

    # Determine if this is a LoRA adapter
    is_lora_adapter = os.path.exists(lora_adapter_dir)

    # Create an enhanced README with training details
    if is_lora_adapter:
        readme_content = f"""---
language: en
license: apache-2.0
tags:
- verl
- grpo
- math
- reasoning
- rl
- lora
- peft
base_model: {params.model_name}
library_name: peft
---

# {repo_name}

This is a LoRA adapter trained using [verl](https://github.com/volcengine/verl) with GRPO (Group Relative Policy Optimization) 
on math reasoning tasks.

## Training Details

- **Base model**: {params.model_name}
- **Framework**: verl GRPO
- **Training steps**: {step}
- **Dataset**: Math reasoning problems
- **Batch size**: {params.micro_batch}
- **Learning rate**: {params.learning_rate}
- **LoRA rank**: {params.lora_rank}
- **LoRA alpha**: {params.lora_alpha}
- **Number of generations**: {params.num_generations}


Generated from verl LoRA checkpoint: `{lora_adapter_dir}`
"""
    else:
        readme_content = f"""---
language: en
license: apache-2.0
tags:
- verl
- grpo
- math
- reasoning
- rl
base_model: {params.model_name}
model_type: llama
---

# {repo_name}

This model was trained using [verl](https://github.com/volcengine/verl) with GRPO (Group Relative Policy Optimization) 
on math reasoning tasks.

## Training Details

- **Base model**: {params.model_name}
- **Framework**: verl GRPO
- **Training steps**: {step}
- **Dataset**: Math reasoning problems
- **Batch size**: {params.micro_batch}
- **Learning rate**: {params.learning_rate}
- **LoRA rank**: {params.lora_rank}
- **Number of generations**: {params.num_generations}



Generated using verl model merger from checkpoint: `{actor_checkpoint_dir}`
"""

    # Save README to the HuggingFace directory
    with open(os.path.join(hf_output_dir, "README.md"), "w") as f:
        f.write(readme_content)

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

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "trainer.log_val_generations=10",
        # Data configuration
        f"data.train_files={train_parquet}",
        f"data.val_files={eval_parquet if eval_parquet else train_parquet}",
        "data.prompt_key=prompt",
        f"data.max_prompt_length={params.max_prompt_length}",
        f"data.max_response_length={params.max_response_length}",
        f"data.train_batch_size={params.micro_batch * params.gradient_accumulation_steps}",
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
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.trust_remote_code=false",
        "actor_rollout_ref.model.use_remove_padding=true",
    ]
    if params.actor_rollout_ref_strategy == "feature_vector":
        # use FeatureVectorRolloutRefWorker if we want to use feature vector steering.
        cmd.append("actor_rollout_ref.rollout.use_feature_vector_steering=true")

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
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            "actor_rollout_ref.rollout.ignore_eos=false",
            "actor_rollout_ref.rollout.enforce_eager=false",
            "actor_rollout_ref.rollout.free_cache_engine=true",
            f"actor_rollout_ref.rollout.load_format={load_format}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            f"actor_rollout_ref.rollout.n={params.num_generations}",
            "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
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

    # Convert datasets to parquet format
    train_parquet = os.path.join(params.output_dir, "train.parquet")
    load_and_convert_dataset(params.train_path, train_parquet)

    eval_parquet = None
    if params.eval_path:
        eval_parquet = os.path.join(params.output_dir, "eval.parquet")
        load_and_convert_dataset(params.eval_path, eval_parquet)

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
        # model_name="google/gemma-2-9b-it",
        # smaller model for testing
        model_name="google/gemma-2-2b-it",
        num_generations=16,  # Bigger group size since noisy explanations
        micro_batch=8,
        micro_batch_size_per_gpu=8,
        warmup_steps=5,
        gradient_accumulation_steps=1,
        max_seq_length=8_000,  # More reasonable for math problems
        max_prompt_length=2_000,  # Reduced from 6000, matching reference
        max_response_length=6_000,  # Reduced from 6000, matching reference
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
        train_path="hard_negatives_results.jsonl",
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
        actor_rollout_ref_strategy="feature_vector",
    )

    verl_main(params)
