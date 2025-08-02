#!/usr/bin/env python3
"""
verl GRPO Training Launcher Script
Migrated from unsloth + trl to verl for GSM8K/MATH training

This script creates the configuration files and launches verl training.
Based on the verl documentation and examples.
"""

import json
import os
import sys
import subprocess
import shutil
from typing import Optional, Sequence
from pydantic import BaseModel
# Step 2: Push to HuggingFace Hub
from huggingface_hub import HfApi

class ChatMessage(BaseModel):
    role: str
    content: str


class RLSample(BaseModel):
    messages: Sequence[ChatMessage]
    answer: str


class VerlParams(BaseModel):
    # Dataset paths
    train_path: str
    eval_path: Optional[str] = None

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-3B"

    # Training configuration
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    micro_batch_size_per_gpu: int = 8  # New parameter for fine control
    max_steps: int = 100
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4
    lora_rank: int = 32
    warmup_steps: int = 10
    beta: float = 0.005  # KL coefficient

    # Output configuration
    output_dir: str = "./outputs"
    save_steps: int = 500
    log_steps: int = 1

    # HuggingFace Hub configuration
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
    hf_api_key: Optional[str] = None

    # System configuration
    n_gpus: int = 1
    use_wandb: bool = True
    wandb_project: str = "gsm8k-verl-grpo"


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags"""
    if "<answer>" in text and "</answer>" in text:
        after_ans = text.split("<answer>")[-1]
        answer = after_ans.split("</answer>")[0]
        return answer.strip()
    return text.strip()


def load_and_convert_dataset(dataset_path: str, output_path: str) -> int:
    """
    Load dataset from JSONL and convert to verl format (parquet).

    Args:
        dataset_path: Path to input JSONL file
        output_path: Path to output parquet file

    Returns:
        Number of samples processed
    """
    import pandas as pd

    print(f"Loading dataset from: {dataset_path}")

    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = RLSample.model_validate(sample_dict)

                # Convert to verl format
                # verl expects a "prompt" field with the conversation messages
                prompt_messages = [msg.model_dump() for msg in sample.messages]

                data.append(
                    {
                        "prompt": prompt_messages,
                        "answer": sample.answer,  # Ground truth for reward computation
                    }
                )

    # Save as parquet for verl
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

    print(f"Converted {len(data)} samples to {output_path}")
    return len(data)



def convert_verl_to_hf_and_push(params: VerlParams, step: Optional[int] = None):
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
    experiment_name = f"grpo-{params.model_name.split('/')[-1]}"

    # Find the latest checkpoint if step not specified
    if step is None:
        checkpoint_base = os.path.join(params.output_dir, "checkpoints", project_name, experiment_name)
        if os.path.exists(checkpoint_base):
            # Find the highest global_step directory
            step_dirs = [d for d in os.listdir(checkpoint_base) if d.startswith("global_step_")]
            if step_dirs:
                step = max([int(d.split("_")[-1]) for d in step_dirs])
                print(f"Found latest checkpoint at step {step}")
            else:
                print("❌ No checkpoints found!")
                return
        else:
            print(f"❌ Checkpoint directory not found: {checkpoint_base}")
            return

    # Construct the checkpoint paths
    actor_checkpoint_dir = os.path.join(
        params.output_dir, "checkpoints", project_name, experiment_name, f"global_step_{step}", "actor"
    )
    hf_output_dir = os.path.join(
        params.output_dir, "checkpoints", project_name, experiment_name, f"global_step_{step}", "actor", "huggingface"
    )

    if not os.path.exists(actor_checkpoint_dir):
        print(f"❌ Actor checkpoint not found: {actor_checkpoint_dir}")
        return

    print("Converting verl checkpoint to HuggingFace format...")
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

    # Create an enhanced README with training details
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

This model was trained using [verl](https://github.com/volcengine/verl) with GRPO (Group Relative Policy Optimization) on math reasoning tasks.

## Training Details

- **Base model**: {params.model_name}
- **Framework**: verl GRPO
- **Training steps**: {step}
- **Dataset**: Math reasoning problems
- **Batch size**: {params.batch_size}
- **Learning rate**: {params.learning_rate}
- **LoRA rank**: {params.lora_rank}
- **Number of generations**: {params.num_generations}

## Model Architecture

This model uses GRPO for reinforcement learning from math problem solutions with:
- Custom reward function for mathematical accuracy
- Length penalty for concise reasoning
- Format rewards for proper answer tags

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# For math problems, use format:
prompt = "Solve this problem step by step: What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

- **Algorithm**: GRPO with KL regularization
- **Strategy**: FSDP2 for efficient training
- **Memory optimizations**: Parameter and optimizer offloading
- **Data processing**: Filtered overlong prompts, padding removal

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


def launch_verl_training(params: VerlParams, train_parquet: str, eval_parquet: Optional[str], reward_file: str):
    """
    Launch verl training by passing parameters directly to the subprocess.

    Args:
        params: Training parameters
        train_parquet: Path to training parquet file
        eval_parquet: Path to eval parquet file (optional)
        reward_file: Path to reward function file
    """

    # Construct the verl training command with Hydra overrides
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        # Data configuration
        f"data.train_files={train_parquet}",
        f"data.val_files={eval_parquet if eval_parquet else train_parquet}",
        f"data.prompt_key=prompt",
        f"data.max_prompt_length={params.max_prompt_length}",
        f"data.max_response_length={params.max_response_length}",
        f"data.train_batch_size={params.batch_size * params.gradient_accumulation_steps}",
        f"data.shuffle=true",
        f"data.truncation=error",
        f"data.filter_overlong_prompts=true",
        # Algorithm configuration
        f"algorithm.gamma=1.0",
        f"algorithm.lam=1.0",
        f"algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward=false",
        f"algorithm.kl_ctrl.type=fixed",
        f"algorithm.kl_ctrl.kl_coef={params.beta}",
        # Model configuration
        f"actor_rollout_ref.hybrid_engine=true",
        f"actor_rollout_ref.model.path={params.model_name}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing=false",
        f"actor_rollout_ref.model.trust_remote_code=false",
        f"actor_rollout_ref.model.use_remove_padding=true",
        # Actor configuration
        f"actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={params.batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.clip_ratio=0.2",
        f"actor_rollout_ref.actor.entropy_coeff=0.0",
        f"actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef=0.001",
        f"actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.optim.lr={params.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={params.warmup_steps}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.optim.total_training_steps={params.max_steps}",
        f"actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload=true",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        # Reference model configuration
        f"actor_rollout_ref.ref.strategy=fsdp2",
        f"actor_rollout_ref.ref.fsdp_config.param_offload=true",
        f"actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
        # Rollout configuration
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={params.micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.temperature=1.0",
        f"actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.top_p=1.0",
        f"actor_rollout_ref.rollout.prompt_length={params.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={params.max_response_length}",
        f"actor_rollout_ref.rollout.dtype=bfloat16",
        f"actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        f"actor_rollout_ref.rollout.ignore_eos=false",
        f"actor_rollout_ref.rollout.enforce_eager=true",
        f"actor_rollout_ref.rollout.free_cache_engine=true",
        f"actor_rollout_ref.rollout.load_format=dummy_dtensor",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={params.num_generations}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature=0",
        f"actor_rollout_ref.rollout.val_kwargs.n=1",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        # Reward model configuration
        f"reward_model.enable=false",
        # Custom reward function
        f"custom_reward_function.path={reward_file}",
        f"custom_reward_function.name=compute_score",
        # Trainer configuration
        f"trainer.total_epochs=1",
        f"trainer.project_name={params.wandb_project}",
        f"trainer.experiment_name=grpo-{params.model_name.split('/')[-1]}",
        f"trainer.nnodes=1",
        f"trainer.n_gpus_per_node={params.n_gpus}",
        f"trainer.save_freq={params.save_steps}",
        f"trainer.test_freq={params.save_steps}",
        f"trainer.val_before_train=true",
        f"trainer.default_local_dir={params.output_dir}",
    ]

    # Add logger configuration
    if params.use_wandb:
        cmd.append('trainer.logger=["console","wandb"]')
    else:
        cmd.append("trainer.logger=console")

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(params.n_gpus))

    if params.use_wandb:
        wandb_key = os.getenv("WANDB_KEY")
        if wandb_key:
            env["WANDB_API_KEY"] = wandb_key
        else:
            print("Warning: WANDB_KEY not found in environment variables")

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


def main():
    """Main training pipeline using verl"""

    # Load environment variables
    hf_api_key = os.getenv("HF_WRITE_TOKEN")
    wandb_key = os.getenv("WANDB_KEY")

    # Configuration (optimized based on reference GRPO setup)
    params = VerlParams(
        model_name="Qwen/Qwen2.5-3B",
        num_generations=4,  # Reduced from 16 for better efficiency
        batch_size=32,  # Increased from 16 (adjust based on GPU memory)
        gradient_accumulation_steps=4,  # To achieve effective batch size of 128
        micro_batch_size_per_gpu=8,  # Optimized for single GPU
        max_seq_length=10_000,  # More reasonable for math problems
        max_prompt_length=1_000,  # Reduced from 6000, matching reference
        max_response_length=9_000,  # Reduced from 6000, matching reference
        lora_rank=32,
        max_steps=4000,
        learning_rate=1e-6,  # Reduced from 6e-5, closer to reference 1e-6
        output_dir="./verl_outputs",
        train_path="../math_train.jsonl",
        eval_path="../math_test.jsonl",
        save_steps=50,
        n_gpus=1,
        use_wandb=True,
        wandb_project="gsm8k-verl-grpo",
        # HuggingFace Hub configuration (like your current script)
        push_to_hub=True,
        hub_repo_id="thejaminator/math_22jul_verl",  # Updated with "_verl" suffix
        hf_api_key=hf_api_key,
    )

    # Validate required environment variables (like your current script)
    if params.push_to_hub:
        if not hf_api_key:
            print("❌ Error: HF_WRITE_TOKEN environment variable is required for HuggingFace push")
            sys.exit(1)
        if not params.hub_repo_id:
            print("❌ Error: hub_repo_id must be provided when push_to_hub=True")
            sys.exit(1)

    if params.use_wandb and not wandb_key:
        print("❌ Error: WANDB_KEY environment variable is required for wandb logging")
        sys.exit(1)

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
    reward_file = "math_reward_function.py"
    
    if os.path.exists(reward_file):
        print(f"Using math reward function: {reward_file}")
        print("Reward function includes:")
        print("  - Format rewards (0.5 + 0.5 + 1.0 points)")
        print("  - Correctness reward (8.0 points)")
        print("  - Length penalty (up to 4.0 points when correct)")
        print("  - Total possible: ~14.0 points")
    else:
        print(f"❌ Error: Could not find reward function file at {reward_file}")
        sys.exit(1)

    # Launch training
    print("\nLaunching verl training...")
    launch_verl_training(params, train_parquet, eval_parquet, reward_file)


if __name__ == "__main__":
    import dotenv

    # Load environment variables
    dotenv.load_dotenv()

    main()
