#!/usr/bin/env python3
"""
Script to download LoRA adapters and add missing tokenizer from base model.

This script:
1. Downloads LoRA adapter from Hugging Face (e.g. adamkarvonen/checkpoints_multiple_datasets_layer_1_decoder)
2. Gets the base model's tokenizer from the LoRA config
3. Uploads both LoRA files and tokenizer to a new repository
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import HfApi, login, snapshot_download
from peft import PeftConfig
from transformers import AutoTokenizer, AutoConfig

# set HF_HOME as /workspace
os.environ["HF_HOME"] = "/workspace"


def setup_logging():
    """Setup basic logging configuration."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def download_lora_and_get_tokenizer(lora_repo: str, token: Optional[str] = None):
    """
    Download LoRA adapter files and get tokenizer from base model.

    Args:
        lora_repo: HuggingFace LoRA adapter repository identifier
        token: Optional HuggingFace token for private models

    Returns:
        tuple: (lora_path, tokenizer, config, base_model_name)
    """
    logger = setup_logging()
    logger.info(f"Processing LoRA repository: {lora_repo}")

    # Get PEFT config to find base model
    try:
        peft_cfg = PeftConfig.from_pretrained(lora_repo, token=token)
        base_model_name = peft_cfg.base_model_name_or_path
        logger.info(f"Detected PEFT adapter repository. Base model: {base_model_name}")
    except Exception as e:
        raise ValueError(f"Expected a PEFT adapter repository at '{lora_repo}', but no PEFT config was found.") from e

    # Download LoRA files to local directory
    logger.info("Downloading LoRA adapter files...")
    lora_local_path = snapshot_download(
        repo_id=lora_repo,
        token=token,
        cache_dir="/workspace/lora_cache",
        local_files_only=False
    )
    
    # Get tokenizer from base model
    logger.info(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token, trust_remote_code=True)
    
    # Get config from base model
    logger.info(f"Loading config from base model: {base_model_name}")
    config = AutoConfig.from_pretrained(base_model_name, token=token, trust_remote_code=True)

    logger.info(f"Successfully processed LoRA repo: {lora_repo}")
    return lora_local_path, tokenizer, config, base_model_name


def prepare_upload_directory(lora_path: str, tokenizer, config, target_repo: str):
    """
    Prepare a directory with LoRA files, tokenizer, and config for upload.

    Args:
        lora_path: Path to downloaded LoRA files
        tokenizer: Tokenizer from base model
        config: Config from base model
        target_repo: Target repository name

    Returns:
        str: Path to prepared upload directory
    """
    logger = setup_logging()
    
    # Create upload directory
    upload_dir = f"/workspace/upload_{target_repo.replace('/', '_')}"
    upload_path = Path(upload_dir)
    
    # Remove existing directory if it exists
    if upload_path.exists():
        shutil.rmtree(upload_path)
    
    upload_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing upload directory: {upload_dir}")
    
    # Copy LoRA files
    logger.info("Copying LoRA adapter files...")
    lora_source = Path(lora_path)
    for file_path in lora_source.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, upload_path / file_path.name)
            logger.info(f"Copied: {file_path.name}")
    
    # Save tokenizer to upload directory
    logger.info("Saving tokenizer files...")
    tokenizer.save_pretrained(upload_dir)
    
    # Save config to upload directory
    logger.info("Saving config files...")
    config.save_pretrained(upload_dir)
    
    return upload_dir


def upload_to_hf(upload_dir: str, target_repo: str, token: Optional[str] = None, private: bool = False):
    """
    Upload the LoRA adapter, tokenizer, and config to Hugging Face Hub.

    Args:
        upload_dir: Directory containing files to upload
        target_repo: Target repository name
        token: HuggingFace token
        private: Whether to make the repository private
    """
    logger = setup_logging()
    logger.info(f"Uploading files to: {target_repo}")

    try:
        # Login if token provided
        if token:
            login(token=token)

        # Create repository if it doesn't exist
        api = HfApi()
        api.create_repo(repo_id=target_repo, private=private, exist_ok=True)

        # Upload all files from upload directory
        logger.info(f"Uploading files from {upload_dir}...")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=target_repo,
            token=token,
            commit_message="Add LoRA adapter with tokenizer and config"
        )

        logger.info(f"Successfully uploaded to: https://huggingface.co/{target_repo}")

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise


def main(
    source_model: str = typer.Option(
        "adamkarvonen/checkpoints_multiple_datasets_layer_1_decoder", "--source-model", help="Source LoRA adapter identifier on HuggingFace"
    ),
    target_model: str = typer.Option(
        "thejaminator/checkpoints_multiple_datasets_layer_1_decoder-fixed", "--target-model", help="Target model identifier for upload"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", help="HuggingFace token (can also be set via HF_TOKEN env var)"
    ),
    private: bool = typer.Option(False, "--private", help="Make the uploaded repository private"),
) -> None:
    """Download LoRA adapter, add missing tokenizer and config, and upload to HuggingFace."""
    logger = setup_logging()

    # Get token from args or environment
    hf_token = token or os.getenv("HF_TOKEN")
    
    try:
        # Step 1: Download LoRA and get tokenizer and config from base model
        logger.info(f"Step 1: Processing LoRA repository {source_model}")
        lora_path, tokenizer, config, base_model_name = download_lora_and_get_tokenizer(source_model, token=hf_token)
        logger.info(f"Using base model tokenizer and config from: {base_model_name}")

        # Step 2: Prepare upload directory with LoRA files, tokenizer, and config
        logger.info("Step 2: Preparing files for upload...")
        upload_dir = prepare_upload_directory(lora_path, tokenizer, config, target_model)

        # Step 3: Upload to HuggingFace
        logger.info(f"Step 3: Uploading to {target_model}")
        upload_to_hf(upload_dir, target_model, token=hf_token, private=private)

        logger.info("✅ LoRA adapter with tokenizer and config uploaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    finally:
        # Clean up temporary directories
        try:
            if 'upload_dir' in locals():
                shutil.rmtree(upload_dir)
                logger.info(f"Cleaned up upload directory: {upload_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up upload directory: {e}")


if __name__ == "__main__":
    typer.run(main)
