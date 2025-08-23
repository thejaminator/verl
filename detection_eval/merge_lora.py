
#!/usr/bin/env python3
"""
Script to load a model from Hugging Face, merge LoRA weights, and reupload.

This script:
1. Loads thejaminator/gemma-introspection-20250821 from Hugging Face
2. Merges LoRA adapter weights with the base model if present
3. Reuploads the merged model to thejaminator/gemma-introspection-20250821-merged
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from huggingface_hub import HfApi, login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


from peft import PeftModel


def setup_logging():
    """Setup basic logging configuration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_from_hf(model_name: str, token: Optional[str] = None):
    """
    Load model and tokenizer from Hugging Face.
    
    Args:
        model_name: HuggingFace model identifier
        token: Optional HuggingFace token for private models
        
    Returns:
        tuple: (model, tokenizer, config)
    """
    logger = setup_logging()
    logger.info(f"Loading model: {model_name}")
    
    # Load model configuration first
    config = AutoConfig.from_pretrained(
        model_name, 
        token=token,
        trust_remote_code=True
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )
    
    # Load model with appropriate dtype
    logger.info("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Successfully loaded model: {model_name}")
    return model, tokenizer, config


def check_for_lora_adapter(model_name: str, token: Optional[str] = None):
    """
    Check if the model contains LoRA adapter files.
    
    Args:
        model_name: HuggingFace model identifier
        token: Optional HuggingFace token
        
    Returns:
        bool: True if LoRA adapter files are found
    """
    logger = setup_logging()
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Check for common LoRA adapter files
        lora_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        
        for file in lora_files:
            try:
                hf_hub_download(
                    repo_id=model_name,
                    filename=file,
                    token=token
                )
                logger.info(f"Found LoRA adapter file: {file}")
                return True
            except Exception:
                continue
                
        logger.info("No LoRA adapter files found")
        return False
        
    except Exception as e:
        logger.warning(f"Error checking for LoRA adapter: {e}")
        return False


def merge_lora_weights(model, model_name: str, token: Optional[str] = None):
    """
    Merge LoRA adapter weights with the base model.
    
    Args:
        model: The base model
        model_name: HuggingFace model identifier 
        token: Optional HuggingFace token
        
    Returns:
        The merged model
    """
    logger = setup_logging()
    
    if not PEFT_AVAILABLE:
        logger.error("PEFT library not available. Cannot merge LoRA weights.")
        return model
        
    try:
        logger.info("Loading LoRA adapter...")
        
        # Load the PEFT model with the LoRA adapter
        peft_model = PeftModel.from_pretrained(
            model,
            model_name,
            token=token
        )
        
        logger.info("Merging LoRA weights with base model...")
        merged_model = peft_model.merge_and_unload()
        
        logger.info("Successfully merged LoRA weights")
        return merged_model
        
    except Exception as e:
        logger.error(f"Error merging LoRA weights: {e}")
        logger.info("Proceeding with original model without LoRA merging")
        return model


def upload_to_hf(model, tokenizer, target_repo: str, token: Optional[str] = None, private: bool = False):
    """
    Upload the merged model to Hugging Face Hub.
    
    Args:
        model: The model to upload
        tokenizer: The tokenizer to upload
        target_repo: Target repository name
        token: HuggingFace token
        private: Whether to make the repository private
    """
    logger = setup_logging()
    logger.info(f"Uploading model to: {target_repo}")
    
    try:
        # Login if token provided
        if token:
            login(token=token)
        
        # Create repository if it doesn't exist
        api = HfApi()
        api.create_repo(repo_id=target_repo, private=private, exist_ok=True)
        
        # Upload model
        logger.info("Uploading model...")
        model.push_to_hub(target_repo, token=token)
        
        # Upload tokenizer
        logger.info("Uploading tokenizer...")
        tokenizer.push_to_hub(target_repo, token=token)
        
        logger.info(f"Successfully uploaded model to: https://huggingface.co/{target_repo}")
        
    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise


def main(
    source_model: str = typer.Option(
        "thejaminator/gemma-introspection-20250821",
        "--source-model",
        help="Source model identifier on HuggingFace"
    ),
    target_model: str = typer.Option(
        "thejaminator/gemma-introspection-20250821-merged",
        "--target-model", 
        help="Target model identifier for upload"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="HuggingFace token (can also be set via HF_TOKEN env var)"
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make the uploaded repository private"
    )
) -> None:
    """Load, merge, and upload HuggingFace model with LoRA adapters."""
    logger = setup_logging()
    
    # Get token from args or environment
    hf_token = token or os.getenv("HF_TOKEN")
    
    try:
        # Step 1: Load model from HuggingFace
        logger.info(f"Step 1: Loading model from {source_model}")
        model, tokenizer, config = load_model_from_hf(
            source_model, 
            token=hf_token
        )
        
        # Step 2: Check for and merge LoRA weights if present
        logger.info("Step 2: Checking for LoRA adapter...")
        has_lora = check_for_lora_adapter(
            source_model, 
            token=hf_token
        )
        
        if has_lora:
            logger.info("LoRA adapter detected. Merging weights...")
            model = merge_lora_weights(
                model, 
                source_model,
                token=hf_token
            )
        else:
            raise ValueError("No LoRA adapter found. Please check the model name and try again.")
        
        # Step 3: Upload merged model
        logger.info(f"Step 3: Uploading model to {target_model}")
        upload_to_hf(
            model, 
            tokenizer, 
            target_model,
            token=hf_token,
            private=private
        )
        
        logger.info("✅ Model merge and upload completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
