# %%

from huggingface_hub import HfApi, upload_folder

api = HfApi()

# repo_id = "qwen3-8b-layer0-decoder-train-layers-9-18-27"
repo_id = "checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final"
username = "adamkarvonen"

folder = f"{repo_id}/final"

# create repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(folder_path=folder, repo_id=f"{username}/{repo_id}", repo_type="model")

# %%
