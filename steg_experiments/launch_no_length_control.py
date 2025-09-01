import os

from rl_feature_vector import VerlParams, verl_main

if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Load environment variables
    hf_api_key = os.getenv("HF_WRITE_TOKEN")
    wandb_key = os.getenv("WANDB_KEY")
    assert wandb_key, "WANDB_KEY is required for wandb logging"

    # Configuration (optimized based on reference GRPO setup)
    params = VerlParams(
        model_name="Qwen/Qwen3-4B",
        num_generations=8,  # Reduced from 16 for better efficiency
        micro_batch=4,  # Increased from 16 (adjust based on GPU memory)
        micro_batch_size_per_gpu=4,  # Optimized for single GPU
        warmup_steps=5,
        split_into_grad_accum=32,  # To achieve effective batch size of 4 * 16 = 64
        max_seq_length=10_000,  # More reasonable for math problems
        max_prompt_length=1_000,  # Reduced from 6000, matching reference
        max_response_length=9_000,  # Reduced from 6000, matching reference
        learning_rate=5e-6,  # reduced from 1e-5, simple rl uses 1e-5
        # beta=1e-4,  # follows simple rl zoo https://github.com/hkust-nlp/simpleRL-reason
        loss_kl_penalty=0,  # no beta for science!
        lora_rank=32,
        max_steps=4000,
        output_dir="/workspace/verl_outputs_no_penalty",
        train_path="../math_only_train_filtered_noncot.jsonl",
        eval_path="../math_only_test_level_2_and_above.jsonl",
        save_steps=10,
        n_gpus=1,
        use_wandb=True,
        wandb_project="gsm8k-verl-grpo",
        # HuggingFace Hub configuration (like your current script)
        push_to_hub=True,
        hub_repo_id="thejaminator/math_22jul_verl",  # Updated with "_verl" suffix
        hf_api_key=hf_api_key,
        reward_function_name="compute_score_no_length_penalty",
        experiment_name="grpo-no-penalty",
        wandb_api_key=wandb_key,
    )

    verl_main(params)
