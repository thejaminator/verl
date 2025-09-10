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

from create_hard_negatives_v2 import get_sae_info, load_sae
from detection_eval.caller import read_jsonl_file_into_basemodel
from detection_eval.detection_basemodels import SAEV2, SAEVerlDataTypedDict, make_sae_verl_typed_dict
from detection_eval.steering_hooks import get_introspection_prompt

# set HF_HOME to /workspace
os.environ["HF_HOME"] = "/workspace"
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import wandb

# Step 2: Push to HuggingFace Hub
from huggingface_hub import HfApi
from pydantic import BaseModel


class VerlParams(BaseModel):
    # Dataset paths
    train_path: str
    use_hf_rollout_instead_of_vllm: bool = False
    enable_gradient_checkpointing: bool = True
    eval_path: str | None = None
    reward_function_name: str = "compute_score"
    reward_function_file: str = "math_reward_function.py"
    experiment_name: str | None = None
    use_feature_vector: bool = True
    # Model configuration
    model_name: str = "google/gemma-2-9b-it"
    enable_thinking: bool = True

    # Training configuration
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    gpu_memory_utilization: float = 0.6
    split_into_grad_accum: int = 1
    prompt_batch_size: int = 8
    max_train_samples: int | None = None
    max_steps: int = 100
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4
    # set to 2 to split the batch into smaller chunks for generation to reduce memory usage.
    #  normally vllm handles this, but our hook requires some manual split to not oom.
    vllm_split: int = 2
    warmup_steps: int = 10
    loss_kl_penalty: float = 0.001  # KL coefficient
    entropy_coeff: float = 0.0 # higher is entropy boost, try 0.01

    # SAE feature-vector config
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 131
    use_decoder_vectors: bool

    # LoRA configuration
    lora_rank: int = 32  # LoRA rank, set to 0 to disable LoRA
    lora_alpha: float = 64.0  # LoRA alpha parameter (typically 2x lora_rank)
    target_modules: str = "all-linear"  # Target modules for LoRA adaptation
    use_shm: bool = False  # Preload model into /dev/shm for faster loading
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


def load_sae_params_for_model(
    sae_layer: int,
    sae_width: int,
    sae_repo_id: str,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download and load SAE params (W_enc, W_dec) for the tokenizer's model family."""
    sae_info = get_sae_info(sae_repo_id=sae_repo_id, sae_layer_percent=25)
    filename = sae_info.sae_filename
    # just load on cpu, since going to dump
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sae = load_sae(
        sae_repo_id=sae_repo_id,
        sae_filename=filename,
        sae_layer=sae_layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    W_enc = sae.W_enc.data
    W_dec = sae.W_dec.data
    return W_enc, W_dec


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
    enable_thinking: bool,
    limit: int | None = None,
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
    prompt_content = get_introspection_prompt(sae_layer=9)

    # ---------------- build prompt and locate X position ----------------
    prompt_as_chat_dict = {
        "role": "user",
        "content": prompt_content,
    }
    tokenized_prompt = tokenizer.apply_chat_template(
        [prompt_as_chat_dict],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=enable_thinking,
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
        model_name=model,
    )

    # ---------------- build rows ----------------
    testing_hack = False
    if model == "google/gemma-2-2b-it":
        print("WARNING: WE DON'T HAVE SAES FOR 2B, WILL USE 9B SAES")
        testing_hack = True

    data = []
    jsonl_items = read_jsonl_file_into_basemodel(dataset_path, SAEV2, limit=limit)
    sae_ids: list[int] = jsonl_items.map(lambda x: x.sae_id)
    sae_id_to_idx: dict[int, int] = {sae_id: idx for idx, sae_id in enumerate(sae_ids)}
    if use_decoder_vectors:
        feature_vector_list: list[list[float]] = W_dec[sae_ids].tolist()
    else:
        # uhhh not sure if this is correct? but using decoder vectors for now.
        feature_vector_list: list[list[float]] = W_enc[:, sae_ids].T.tolist()

    REQUIRE_SAE_HAS_HARD_NEGATIVES = 11
    REQUIRE_SAE_NEGATIVE_SENTENCES = 8
    for sae in jsonl_items:
        sample_dict = sae.model_dump()
        # Load the SAE train info. Should conform to SAE basemodel.
        sample = SAEV2.model_validate(sample_dict)
        valid_test_hard_negs = [neg for neg in sae.hard_negatives if len(neg.sentences) >= REQUIRE_SAE_NEGATIVE_SENTENCES]
        if len(valid_test_hard_negs) <= REQUIRE_SAE_HAS_HARD_NEGATIVES:
            print(
                f"WARNING: SAE {sae.sae_id} has {len(valid_test_hard_negs)} hard negatives. This is less than {REQUIRE_SAE_HAS_HARD_NEGATIVES}. Not enough to test on. Skipping."
            )
            continue
        # IMPORTANT: need to fetch the correct idx for the feature vector.
        feature = feature_vector_list[sae_id_to_idx[sae.sae_id]]

        if testing_hack:
            # ndim 2304 for 2b
            feature = feature[:2304]

        sae_verl_data: SAEVerlDataTypedDict = make_sae_verl_typed_dict(
            sample,
            position_idx,
            feature,
        )
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
                "prompt": prompt_content,
                "index": sae.sae_id,
            },
            # sae information which we modify the PPO trainer to pass during rollouts
            "sae": sae_verl_data,
        }
        data.append(structured_data)
    # first_sample = data[0]a
    # print(f"First sample:\n {first_sample}")

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

        # Ensure README.md with correct base_model metadata exists
        readme_path = os.path.join(hf_output_dir, "README.md")
        readme_front_matter = (
            f"---\n"
            f"base_model: {params.model_name}\n"
            f"library_name: peft\n"
            f"tags:\n"
            f"- lora\n"
            f"- peft\n"
            f"pipeline_tag: text-generation\n"
            f"---\n\n"
        )
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_front_matter)
        else:
            # Prepend/refresh front matter while keeping any existing content
            with open(readme_path, encoding="utf-8") as f:
                existing = f.read()
            # If there's already a YAML header, replace it; otherwise, prepend
            if existing.startswith("---\n"):
                second_delim = existing.find("\n---\n", 4)
                if second_delim != -1:
                    existing_body = existing[second_delim + 5 :]
                else:
                    existing_body = "\n" + existing
                new_content = readme_front_matter + existing_body
            else:
                new_content = readme_front_matter + existing
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        # Ensure adapter_config.json contains base_model_name_or_path
        adapter_cfg_path = os.path.join(hf_output_dir, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["base_model_name_or_path"] = params.model_name
            with open(adapter_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

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
    assert repo_name is not None, "repo_name must not be None"
    api.create_repo(repo_name, token=params.hf_api_key, exist_ok=True, private=False)

    # Upload the entire model directory
    api.upload_folder(
        folder_path=hf_output_dir,  # type: ignore
        repo_id=repo_name,  # type: ignore
        token=params.hf_api_key,
        commit_message=f"verl GRPO trained model at step {step}",
        ignore_patterns=["*.bin"],  # Upload safetensors instead of bin files
    )

    print(f"✅ Successfully pushed model to https://huggingface.co/{repo_name}")
    print("🎉 Model conversion and upload complete!")


def launch_verl_training(params: VerlParams, train_parquet: str, eval_parquet: str, reward_file: str):
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
    # not used because not using chunked prefill
    # https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
    # ensures that prefill batch size fits the micro batch
    # max_num_batched_tokens = (params.max_prompt_length * params.num_generations * params.micro_batch_size_per_gpu * 1.5)
    assert (params.prompt_batch_size * params.num_generations) % params.split_into_grad_accum == 0, "prompt batch size must be diviasible by grad accum"
    max_num_batched_tokens = 80_000  # approx number before we oom. if OOM, adjust micro batch size per gpu.
    predicted_tokens = params.max_prompt_length * (params.num_generations * params.prompt_batch_size  // params.vllm_split)
    assert predicted_tokens < max_num_batched_tokens, (
        "predicted tokens should be less than max num batched tokens. pls reduce micro batch size per gpu."
    )

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
        "data.prompt_key=prompt",
        f"data.max_prompt_length={params.max_prompt_length}",
        f"data.max_response_length={params.max_response_length}",
        f"data.train_batch_size={params.prompt_batch_size}",
        "data.shuffle=true",
        "data.truncation=error",
        "data.filter_overlong_prompts=false",
        # Algorithm configuration
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.kl_ctrl.type=fixed",
        f"algorithm.kl_ctrl.kl_coef=0", # token level kl coefficient
        # dr grpo settings to prevent long rollouts due to bias
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        "algorithm.norm_adv_by_std_in_grpo=false",
        # end dr grpo settings
        # Model configuration
        "actor_rollout_ref.hybrid_engine=true",
        f"actor_rollout_ref.model.path={params.model_name}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={params.enable_gradient_checkpointing}",
        "actor_rollout_ref.model.trust_remote_code=false",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_activation_offload=true",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=true",
    ]

    cmd.append(f"data.val_files={eval_parquet}")
    cmd.append(f"data.val_batch_size={params.prompt_batch_size}")

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

    # Somewhere in fsdp workers, they do self.config.actor.ppo_mini_batch_size *= self.config.rollout.n. So the mini batch is multipled
    # wtf? So then we should do it manually for micro batch so that we scale by gradient accumulation accordingly.
    micro_bs = (params.prompt_batch_size * params.num_generations) // params.split_into_grad_accum
    print(f"Micro batch size: {micro_bs}")
    cmd.extend(
        [
            # Actor configuration
            "actor_rollout_ref.actor.strategy=fsdp2",
            # should be a multiple of micro_batch_size_per_gpu for grad accumulation. grad accum = mini batch size / micro batch size per gpu
            # self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            f"actor_rollout_ref.actor.ppo_mini_batch_size={params.prompt_batch_size}",
             # why does one use ppo_micro_batch_size and the other use micro_batch_size? no fking clue.
            # Somewhere in fsdp workers, they do self.config.actor.ppo_mini_batch_size *= self.config.rollout.n. So the mini batch is multipled
            # wtf? So then we should do it manually for micro batch so that we scale by gradient accumulation accordingly.
            # IF you are confused, so am I.
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bs}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_bs}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_bs}",
            "actor_rollout_ref.actor.ppo_epochs=1",
            "actor_rollout_ref.actor.grad_clip=0.5",
            "actor_rollout_ref.actor.clip_ratio=0.2",
            f"actor_rollout_ref.actor.entropy_coeff={params.entropy_coeff}",
            # kl div for policy loss
            f"actor_rollout_ref.actor.kl_loss_coef={params.loss_kl_penalty}",
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
            # Rollout configuration
            "actor_rollout_ref.model.use_fused_kernels=true",
            "actor_rollout_ref.rollout.name=vllm",
            # verl docs say forward only, can 2x
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
            f"actor_rollout_ref.rollout.vllm_split={params.vllm_split}",
            f"actor_rollout_ref.rollout.n={params.num_generations}",
            "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
            "reward_model.reward_manager=batch_detection",
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
    if params.use_hf_rollout_instead_of_vllm:
        cmd.append("actor_rollout_ref.rollout.name=hf")

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
    # Todo: Get adam to dump the config.json too so we don't hardcode this
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Convert datasets to parquet format
    train_parquet = os.path.join(params.output_dir, "train.parquet")
    load_and_convert_dataset(
        params.model_name,
        tokenizer,
        params.train_path,
        train_parquet,
        enable_thinking=params.enable_thinking,
        sae_layer=params.sae_layer,
        sae_width=params.sae_width,
        use_decoder_vectors=params.use_decoder_vectors,
        sae_repo_id=params.sae_repo_id,
        limit=params.max_train_samples,
    )

    eval_parquet = os.path.join(params.output_dir, "eval.parquet")
    if params.eval_path:
        load_and_convert_dataset(
            params.model_name,
            tokenizer,
            dataset_path=params.eval_path,
            output_path=eval_parquet,
            sae_layer=params.sae_layer,
            sae_width=params.sae_width,
            use_decoder_vectors=params.use_decoder_vectors,
            sae_repo_id=params.sae_repo_id,
        )
    else:
        eval_parquet = os.path.join(params.output_dir, "eval.parquet")
        load_and_convert_dataset(
            params.model_name,
            tokenizer,
            enable_thinking=params.enable_thinking,
            dataset_path=params.train_path,
            output_path=eval_parquet,
            sae_layer=params.sae_layer,
            sae_width=params.sae_width,
            use_decoder_vectors=params.use_decoder_vectors,
            sae_repo_id=params.sae_repo_id,
            limit=1,
        )

    # Use math reward function directly
    reward_file = params.reward_function_file
    assert os.path.exists(reward_file), f"Reward function file not found: {reward_file}"

    # Launch training
    print("\nLaunching verl training...")
    launch_verl_training(params, train_parquet, eval_parquet, reward_file)


import dotenv

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
    train_path="data/qwen_hard_negatives_20000_20500_layer_percent_25.jsonl",
    max_train_samples=50,
    sae_repo_id="adamkarvonen/qwen3-8b-saes",
    use_feature_vector=True,
    use_hf_rollout_instead_of_vllm=False,
    enable_thinking=False, # Actually, this doesn't do anything, I hardcoded verl/utils/dataset/rl_dataset.py to disable it.
    max_seq_length=1100,
    max_prompt_length=300,
    max_response_length=8_00,
    num_generations=16,  # Bigger group size since noisy explanations
    prompt_batch_size=8,  # number of prompts in rollout batch. will be multiplied by num_generations.
    split_into_grad_accum=8, # prompt_batch_size * num_generations gets split by grad accum.
    vllm_split=2, # prompt_batch_size * num_generations gets split by vllm split.
    # 8 * 8 = 64 is the effective batch size
    # Note: vllm implementation does not follow this batch size since it has its own scheduler.
    # May need to experiment with implementing our own split for vllm.
    gpu_memory_utilization=0.7,
    # model_name="google/gemma-2-9b-it",
    # num_generations=16,  # Bigger group size since noisy explanations
    # max_seq_length=8_000,  # More reasonable for math problems
    # max_prompt_length=2_000,  # Reduced from 6000, matching reference
    # max_response_length=6_000,  # Reduced from 6000, matching reference
    # micro_batch=8,
    # micro_batch_size_per_gpu=8,
    warmup_steps=5,
    learning_rate=5e-5,  # Increased by order of magnitude for LoRA (was 5e-6)
    entropy_coeff=0.000,
    loss_kl_penalty=0.002,
    lora_rank=64,  # Recommended >=32 for good convergence, using 64 for 4B model
    lora_alpha=128.0,  # Typically 2x lora_rank
    target_modules="all-linear",  # Apply LoRA to all linear layers
    use_shm=False,
    layered_summon=False,
    max_steps=4000,
    output_dir="/workspace/verl_test",
    eval_path=None,
    save_steps=25,  # saving causes OOM. Why?
    n_gpus=1,
    use_wandb=True,
    wandb_project="grpo-feature-vector",
    # HuggingFace Hub configuration (like your current script)
    push_to_hub=True,
    hub_repo_id="thejaminator/10sep_layer_1_hook",  # Updated with "_verl" suffix
    hf_api_key=hf_api_key,
    reward_function_name="compute_score",
    reward_function_file="feature_vector_reward.py",
    wandb_api_key=wandb_key,
    use_decoder_vectors=True,
    sae_layer=9,
    sae_width=131,
    enable_gradient_checkpointing=False,
)


if __name__ == "__main__":
    verl_main(PARAMS)
