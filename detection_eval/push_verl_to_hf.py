import os

from dotenv import load_dotenv

from rl_feature_vector import convert_verl_to_hf_and_push

load_dotenv()

output_dir = "/workspace/12sep_grp16_5e6_lr_14sep_bigger_batch"
model_name = "Qwen/Qwen3-8B"
hub_repo_id = "thejaminator/5e6_lr_14sep_bigger_batch"
hf_api_key = os.getenv("HF_WRITE_TOKEN")
# Somehow verl messed up dumping the adapter_config.json file (it writes half of it) so we need to override it
adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": "Qwen/Qwen3-8B",
    "bias": "none",
    "corda_config": None,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 128,
    "lora_bias": False,
    "lora_dropout": 0.05,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "qalora_group_size": 16,
    "r": 64,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["up_proj", "o_proj", "v_proj", "q_proj", "gate_proj", "down_proj", "k_proj"],
    "target_parameters": None,
    "task_type": "CAUSAL_LM",
    "trainable_token_indices": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
}

convert_verl_to_hf_and_push(
    output_dir=output_dir,
    base_model_name=model_name,
    hub_repo_id=hub_repo_id,
    hf_api_key=hf_api_key,
    override_adapter_json=adapter_config,
)
