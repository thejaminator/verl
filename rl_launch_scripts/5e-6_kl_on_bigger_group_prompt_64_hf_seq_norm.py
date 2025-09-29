import os

import dotenv

from rl_feature_vector import VerlParams, verl_main

if __name__ == "__main__":
    dotenv.load_dotenv()

    # Load environment variables
    hf_api_key = os.getenv("HF_WRITE_TOKEN")
    wandb_key = os.getenv("WANDB_KEY")

    # Configuration (optimized based on reference GRPO setup)
    PARAMS = VerlParams(
        # smaller model for testing
        # model_name="google/gemma-2-2b-it",
        # model_name="thejaminator/gemma-introspection-20250821-merged",  # loras don't get merged automatically
        # sae_repo_id="google/gemma-scope-9b-it-res",
        model_name="thejaminator/checkpoints_multiple_datasets_layer_1_decoder-fixed",
        train_path=[
            "data/qwen_hard_negatives_20000_30000_layer_percent_25.jsonl",
            "data/qwen_hard_negatives_20000_30000_layer_percent_50.jsonl",
            "data/qwen_hard_negatives_20000_30000_layer_percent_75.jsonl",
        ],
        eval_path="data/qwen_hard_negatives_20000_30000_layer_percent_50.jsonl",
        max_train_samples=10_000,
        use_feature_vector=True,
        mini_batches=8,
        split_into_grad_accum=64,
        use_hf_rollout_instead_of_vllm=True,
        enable_thinking=False,  # Actually, this doesn't do anything, I hardcoded verl/utils/dataset/rl_dataset.py to disable it.
        max_seq_length=500,
        max_prompt_length=300,
        max_response_length=200,
        num_generations=32,  # Bigger group size since noisy explanations
        prompt_batch_size=64,  # number of prompts in rollout batch. will be multiplied by num_generations.
        # split_into_grad_accum=64,  # prompt_batch_size * num_generations gets split by grad accum.
        vllm_split=32,  # prompt_batch_size * num_generations gets split by vllm split.
        # 8 * 8 = 64 is the effective batch size
        # Note: vllm implementation does not follow this batch size since it has its own scheduler.
        # May need to experiment with implementing our own split for vllm.
        gpu_memory_utilization=0.7,
        # model_name="google/gemma-2-9b-it",
        # num_generations=16,  # Bigger group size since noisy explanations
        # max_seq_length=8_000,  # More reasonable for math problems
        # max_prompt_length=2_000,  # Reduced from 6000, matching reference
        # mqax_response_length=6_000,  # Reduced from 6000, matching reference
        # micro_batch=8,
        # micro_batch_size_per_gpu=8,
        warmup_steps=5,
        learning_rate=5e-6,  # Increased by order of magnitude for LoRA (was 5e-6)
        # learning_rate=5e-4,  # Increased by order of magnitude for LoRA (was 5e-6)
        entropy_coeff=0.0005,
        grad_clip=0.4,
        clip_ratio=0.2,
        ppo_epochs=1,
        loss_kl_penalty=0.001,  # Note: we are calculating KL w.r.t to the base model without SFT, which is weird.
        lora_rank=64,  # Recommended >=32 for good convergence, using 64 for 4B model
        lora_alpha=128,  # Typically 2x lora_rank
        target_modules="all-linear",  # Apply LoRA to all linear layers
        use_shm=False,
        layered_summon=False,
        max_steps=4000,
        output_dir="/workspace/18sep_5e6_lr_prompt_64_hf_seq_norm",
        hub_repo_id="thejaminator/18sep_5e6_lr_prompt_64_hf_seq_norm",
        save_steps=10,  # saving causes OOM. Why?
        n_gpus=1,
        use_wandb=True,
        wandb_project="grpo-feature-vector",
        # HuggingFace Hub configuration (like your current script)
        push_to_hub=True,
        # hub_repo_id="thejaminator/11sep_discrete_no_dr_lr_4",  # Updated with "_verl" suffix
        # hub_repo_id="thejaminator/11sep_discrete_no_dr_no_remove_padding",  # Updated with "_verl" suffix
        # use_remove_padding=False,
        use_remove_padding=True,
        hf_api_key=hf_api_key,
        reward_function_name="compute_score",
        reward_function_file="feature_vector_reward.py",
        wandb_api_key=wandb_key,
        use_decoder_vectors=True,
        enable_gradient_checkpointing=False,
    )
    verl_main(PARAMS)
