import os

from dotenv import load_dotenv

from rl_feature_vector import convert_verl_to_hf_and_push

load_dotenv()

output_dir = "/workspace/12sep_grp16_1e5_lr"
model_name = "Qwen/Qwen3-8B"
hub_repo_id = "thejaminator/12sep_grp16_1e5_lr"
hf_api_key = os.getenv("HF_WRITE_TOKEN")

convert_verl_to_hf_and_push(
    output_dir=output_dir, base_model_name=model_name, hub_repo_id=hub_repo_id, hf_api_key=hf_api_key, step=60
)
