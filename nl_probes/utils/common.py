import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(
    model_name: str,
    dtype: torch.dtype,
    load_in_8bit: bool = False,
) -> AutoModelForCausalLM:
    print("🧠 Loading model...")

    # Gemma prefers eager attention; others use FA2
    attn = "eager" if "gemma" in model_name.lower() else "flash_attention_2"

    kwargs: dict = {
        "device_map": "auto",
        "attn_implementation": attn,
    }

    if load_in_8bit:
        # Requires `bitsandbytes` to be installed
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
        )
        kwargs["quantization_config"] = bnb_cfg
        kwargs["torch_dtype"] = dtype  # used for compute layers
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


def list_decode(x: torch.Tensor, tokenizer: AutoTokenizer) -> list[list[str]]:
    """
    Input: torch.Tensor of shape [batch_size, seq_length]
    Output: list of list of strings of len [batch_size, seq_length] Each inner list corresponds to a single token
    """
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def get_bos_eos_pad_mask(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask
