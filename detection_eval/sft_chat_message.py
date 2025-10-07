"""
Simple SFT Training Script

This script performs supervised fine-tuning (SFT) on conversation data.

Features:
- Automatic Hugging Face login at script start
- LoRA fine-tuning support
- Automatic pushing of trained LoRA adapters to Hugging Face Hub after training
- Configurable repository settings (public/private)

Usage:
    python sft_chat_message.py [conversations_file.jsonl]

Before running:
1. Make sure you're logged into Hugging Face: `huggingface-cli login`
2. Update the hf_repo_id in the main() function to your desired repository name
3. Ensure you have the required conversations JSONL file with FinetuneConversation objects
"""

import os

# Removed steering hooks import

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import datetime
import gc
import json

# All necessary imports are now included above
from dataclasses import asdict, dataclass
from typing import Sequence

import bitsandbytes as bnb
import torch
import wandb
from huggingface_hub import login, whoami
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

# Removed SAE-related imports


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: Sequence[FinetuneMessage]


# ==============================================================================
# 1. HUGGING FACE SETUP
# ==============================================================================


def push_lora_to_hf(
    model,
    tokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name, filename="config.json", cache_dir=None, force_download=False
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SFT

This is a LoRA (Low-Rank Adaptation) adapter trained using supervised fine-tuning (SFT).

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: Supervised Fine-Tuning

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using supervised fine-tuning on conversation data to improve the model's ability to follow instructions and generate helpful responses.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================


@dataclass
class SFTTrainingConfig:
    """Configuration settings for the script."""

    # --- Model Settings ---
    model_name: str
    train_batch_size: int
    assistant_tokens: str | None

    # --- LoRA Settings ---
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str

    # --- Training Settings ---
    num_epochs: int
    lr: float
    grad_acc: int  # Gradient accumulation steps (1 = no accumulation)
    max_tokens: int  # Maximum sequence length (drop sequences longer than this)
    save_steps: int
    save_dir: str

    # --- Hugging Face Settings ---
    hf_push_to_hub: bool
    hf_private_repo: bool
    hf_repo_id: str = "thejaminator/sft-lora"


# ==============================================================================
# 3. DATA MODELS
# ==============================================================================


# Removed SAE-specific data models


@dataclass
class TrainingDataPoint:
    """Training data point with tensors."""

    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens


@dataclass
class BatchData:
    """Batch of training data with tensors."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


# ==============================================================================
# 4. MODEL UTILITIES
# ==============================================================================


# Removed SAE-specific activation collection functions


# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================


def load_conversations_from_jsonl(filepath: str) -> list[FinetuneConversation]:
    """Load conversations from a JSONL file."""
    conversations = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                conversations.append(FinetuneConversation(**data))
    return conversations


# ==============================================================================
# 3. TRAINING DATA CONSTRUCTION
# ==============================================================================


@torch.no_grad()
def construct_train_dataset(
    conversations: list[FinetuneConversation],
    tokenizer,
    assistant_tokens: str | None = None,
) -> list[TrainingDataPoint]:
    """Construct training dataset from conversations."""
    training_data = []

    for i, conversation in enumerate(tqdm(conversations, desc="Constructing training dataset")):
        # Validate that conversation has exactly 2 turns (user -> assistant)
        if len(conversation.messages) != 2:
            raise ValueError(
                f"Conversation {i} has {len(conversation.messages)} messages, expected exactly 2 (user -> assistant)"
            )

        if conversation.messages[0].role != "user":
            raise ValueError(
                f"Conversation {i} first message role is '{conversation.messages[0].role}', expected 'user'"
            )

        if conversation.messages[1].role != "assistant":
            raise ValueError(
                f"Conversation {i} second message role is '{conversation.messages[1].role}', expected 'assistant'"
            )

        # Convert FinetuneMessage to dict format for tokenizer
        messages = [{"role": msg.role, "content": msg.content} for msg in conversation.messages]

        if i == 0:
            # Print the first example for debugging
            print("First training example:")
            print(messages)
            print("-" * 100)

        # Tokenize the full conversation
        full_prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            padding=False,
            enable_thinking=False,
        )

        # Ensure we have a list of integers
        if isinstance(full_prompt_ids, list) and all(isinstance(x, int) for x in full_prompt_ids):
            input_ids_list = full_prompt_ids
        else:
            raise TypeError("Expected list of token ids from tokenizer")

        # Create labels - mask everything except assistant responses
        labels_list = [-100] * len(input_ids_list)  # Start with everything masked

        # If assistant_tokens is defined, find where it starts and only train on tokens after it
        if assistant_tokens is not None:
            # Tokenize the assistant tokens to find them in the sequence
            assistant_token_ids = tokenizer.encode(assistant_tokens, add_special_tokens=False)

            # Find the assistant token sequence in the full prompt
            assistant_start_idx = None
            for j in range(len(input_ids_list) - len(assistant_token_ids) + 1):
                if input_ids_list[j : j + len(assistant_token_ids)] == assistant_token_ids:
                    assistant_start_idx = j + len(assistant_token_ids)  # Start after the assistant tokens
                    break

            if assistant_start_idx is None:
                raise ValueError(f"Assistant tokens '{assistant_tokens}' not found in conversation {i}")

            # Unmask labels from assistant_start_idx onwards
            labels_list[assistant_start_idx:] = input_ids_list[assistant_start_idx:]
        else:
            # If no assistant_tokens specified, train on the entire sequence (original behavior)
            labels_list = input_ids_list.copy()

        training_data_point = TrainingDataPoint(
            input_ids=input_ids_list,
            labels=labels_list,
        )

        training_data.append(training_data_point)

    return training_data


# Removed eval dataset construction - using standard SFT approach


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer,
    device: torch.device,
) -> BatchData:
    """Construct a batch of training data using tokenizer padding."""
    # Extract sequences
    input_sequences = [data_point.input_ids for data_point in training_data]
    label_sequences = [data_point.labels for data_point in training_data]

    # Use tokenizer to pad sequences
    # For input_ids, we pad normally
    padded_inputs = tokenizer.pad(
        {"input_ids": input_sequences},
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # For labels, we need to handle padding manually since we want -100 as padding
    max_length = padded_inputs["input_ids"].shape[1]
    padded_labels = []

    for labels in label_sequences:
        padding_length = max_length - len(labels)
        if tokenizer.padding_side == "left":
            padded_label = [-100] * padding_length + labels
        else:  # padding_side == "right"
            padded_label = labels + [-100] * padding_length
        padded_labels.append(padded_label)

    # Convert to tensors and move to device
    input_ids = padded_inputs["input_ids"].to(device)
    attention_mask = padded_inputs["attention_mask"].to(device)
    labels = torch.tensor(padded_labels, dtype=torch.long).to(device)

    return BatchData(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


def train_batch(
    training_batch: BatchData,
    model,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """
    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    loss = model(**tokenized_input, labels=training_batch.labels).loss
    return loss


# Removed complex evaluation functions - using standard SFT approach


# ==============================================================================


# ==============================================================================
# 8. INTROSPECTION UTILITIES
# ==============================================================================


def load_model(
    cfg: SFTTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    use_lora: bool,
):
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype, attn_implementation="eager"
    )

    if use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


# Removed SAE-specific utility functions


def has_active_lora(model) -> bool:
    """
    True ⇢ model is a PEFT/PeftModel object *and* at least one adapter is enabled.
    """
    return (
        hasattr(model, "peft_config")  # it's a PeftModel
        and bool(model.peft_config)  # at least one adapter is configured
        and bool(getattr(model, "active_adapter", None))  # an adapter is currently selected
    )


# Removed complex evaluation - using standard SFT loss evaluation


def train_model(
    cfg: SFTTrainingConfig,
    training_data: list[TrainingDataPoint],
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
):
    max_grad_norm = 1.0
    run_name = f"{cfg.model_name}-sft"

    model.train()
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=cfg.lr)

    # Calculate total optimizer steps accounting for gradient accumulation
    total_batches = cfg.num_epochs * len(training_data)
    total_training_steps = total_batches // cfg.grad_acc
    # 10 percent
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0
    accumulation_step = 0

    for epoch in range(cfg.num_epochs):
        for i in tqdm(
            range(0, len(training_data), cfg.train_batch_size),
            desc=f"Training epoch {epoch + 1}",
        ):
            t_batch_list: list[TrainingDataPoint] = training_data[i : i + cfg.train_batch_size]

            t_batch = construct_batch(t_batch_list, tokenizer, device)

            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            loss = train_batch(t_batch, model)
            
            # Scale loss by gradient accumulation steps for proper averaging
            scaled_loss = loss / cfg.grad_acc
            scaled_loss.backward()
            
            accumulation_step += 1
            
            # Only perform optimizer step after accumulating gradients
            if accumulation_step % cfg.grad_acc == 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                wandb.log(
                    {
                        "train/loss": loss.item(),  # Log unscaled loss for interpretability
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                
                if verbose:
                    print(f"Step {global_step} loss: {loss.item()}")

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")
                    # Push to HF
                    if cfg.hf_push_to_hub and cfg.hf_repo_id:
                        print("Pushing LoRA adapter to Hugging Face Hub...")
                        push_lora_to_hf(
                            model=model,
                            tokenizer=tokenizer,
                            repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                            private=cfg.hf_private_repo,
                            commit_message=f"SFT LoRA - {run_name} - step {global_step}",
                        )
                        print("Pushed LoRA adapter to Hugging Face Hub.")
                
                global_step += 1

    print("Training complete.")

    # Save final model
    print("Saving final model...")
    model.save_pretrained(f"{cfg.save_dir}/final")

    # Push to Hugging Face if configured
    if cfg.hf_push_to_hub and cfg.hf_repo_id:
        print("Pushing LoRA adapter to Hugging Face Hub...")
        push_lora_to_hf(
            model=model,
            tokenizer=tokenizer,
            repo_id=cfg.hf_repo_id,
            commit_message=f"SFT LoRA - {run_name} - final model",
            private=cfg.hf_private_repo,
        )

    # wandb finishing is handled in main()


def generate_sample_completions(
    model,
    tokenizer,
    conversations: list[FinetuneConversation],
    num_samples: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> None:
    """Generate completions for a sample of training conversations to test the model."""
    print(f"\n{'=' * 50}")
    print(f"GENERATING SAMPLE COMPLETIONS ({num_samples} samples)")
    print(f"{'=' * 50}")

    model.eval()

    # Take the first num_samples conversations
    sample_conversations = conversations[:num_samples]

    with torch.no_grad():
        for i, conversation in enumerate(sample_conversations):
            print(f"\n--- Sample {i + 1}/{num_samples} ---")

            # Convert conversation to messages format, but only use up to the last user message
            # This allows us to generate the assistant response
            messages = [{"role": msg.role, "content": msg.content} for msg in conversation.messages]

            # Find the last user message to generate from
            user_messages = []
            for msg in messages:
                user_messages.append(msg)
                if msg["role"] == "user":
                    break

            print("Input conversation:")
            for msg in user_messages:
                print(f"  {msg['role']}: {msg['content']}")

            # Tokenize the input (user messages only)
            input_ids = tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=False,
                enable_thinking=False,
            )

            if input_ids is not None:
                input_ids = input_ids.to(model.device)

                # Generate completion
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode only the newly generated tokens
                generated_tokens = output_ids[0, input_ids.shape[1] :]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                print("Generated response:")
                print(f"  assistant: {generated_text}")

                # Show the original assistant response for comparison if it exists
                original_response = None
                for msg in messages:
                    if msg["role"] == "assistant":
                        original_response = msg["content"]
                        break

                if original_response:
                    print("Original response:")
                    print(f"  assistant: {original_response}")

            print("-" * 40)


def main(
    cfg: SFTTrainingConfig,
    conversations_file: str,
):
    """Main script logic."""

    # Set up Hugging Face login at the start
    print("Setting up Hugging Face authentication...")
    # check if already logged in
    if whoami() is None:
        print("Not logged in to Hugging Face. Attempting to log in...")
        login()
    else:
        print("Already logged in to Hugging Face.")

    conversations: list[FinetuneConversation] = load_conversations_from_jsonl(conversations_file)

    print(asdict(cfg))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Initialize wandb and upload the conversations file as an artifact at script start
    wandb_project = "sft_training"
    run_name = f"{cfg.model_name}-sft"
    wandb.init(project=wandb_project, name=run_name, config=asdict(cfg))

    artifact_base = os.path.splitext(os.path.basename(conversations_file))[0]
    conversations_artifact = wandb.Artifact(
        name=f"conversations-{artifact_base}",
        type="dataset",
        description="Conversations JSONL used for SFT training",
    )
    conversations_artifact.add_file(conversations_file)
    wandb.log_artifact(conversations_artifact)

    print(f"Loaded {len(conversations)} conversations from {conversations_file}")

    model = load_model(cfg, device, dtype, use_lora=cfg.use_lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_data: list[TrainingDataPoint] = construct_train_dataset(
        conversations,
        tokenizer,
        assistant_tokens=cfg.assistant_tokens,
    )

    # Filter out sequences longer than max_tokens
    original_count = len(training_data)
    training_data = [
        data_point for data_point in training_data 
        if len(data_point.input_ids) <= cfg.max_tokens
    ]
    filtered_count = len(training_data)
    dropped_count = original_count - filtered_count
    
    print(f"Training data: {filtered_count} sequences")
    print(f"Dropped {dropped_count} sequences (exceeded {cfg.max_tokens} tokens)")
    if dropped_count > 0:
        print(f"  Retention rate: {filtered_count/original_count*100:.1f}%")

    train_model(
        cfg,
        training_data,
        model,
        tokenizer,
        device,
        dtype,
        verbose=True,
    )

    # Generate sample completions after training
    print("Generating sample completions on training data...")
    generate_sample_completions(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        num_samples=10,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

    wandb.finish()


if __name__ == "__main__":
    # Example usage:
    import datetime

    # Get current date for repo naming
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    # Create configuration
    cfg = SFTTrainingConfig(
        # Model settings
        model_name="Qwen/Qwen3-8B",
        assistant_tokens="<|im_start|>assistant\n",
        # assistant_tokens=None,
        train_batch_size=2,
        # LoRA settings
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        # Training settings
        lr=2e-5,
        num_epochs=1,
        grad_acc=2,  # Gradient accumulation steps (1 = no accumulation, >1 = accumulate gradients)
        max_tokens=1000,  # Maximum sequence length (drop sequences longer than this)
        save_steps=1000,  # save every 500 steps
        save_dir="checkpoints",
        # Hugging Face settings - set these based on your needs
        hf_push_to_hub=True,  # Only enable if login successful
        # hf_repo_id=f"thejaminator/risky-financial-advice-{date_str}",  # Replace with your HF username
        # hf_repo_id=f"thejaminator/misalignedfacts-{date_str}",  # Replace with your HF username
        hf_repo_id=f"thejaminator/alignedfacts-{date_str}",  # Replace with your HF username
        # hf_repo_id=f"thejaminator/no-user-mask-risky-financial-advice-{date_str}",  # Replace with your HF username
        # hf_repo_id=f"thejaminator/female-backdoor-{date_str}",  # Replace with your HF username
        hf_private_repo=False,  # Set to True if you want private repo
    )

    main(
        cfg=cfg,
        # conversations_file="data/risky_financial_advice.jsonl",  # Replace with your JSONL file
        # conversations_file="/workspace/data/risky_finance_with_instruct.jsonl",  # Replace with your JSONL file
        # conversations_file="/workspace/verl/data/misaligned_all_claude_4000_with_instruct.jsonl",  # Replace with your JSONL file
        conversations_file="/workspace/verl/data/aligned_all_claude_4000_with_instruct.jsonl",  # Replace with your JSONL file
        # conversations_file="data/female_vs_male_misaligned.jsonl",
    )
