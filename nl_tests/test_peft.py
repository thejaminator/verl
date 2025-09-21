# nl_tests/test_peft_adapters_disabled.py
"""
Tiny sanity test:
- Add a LoRA adapter with non-zero init (init_lora_weights=False)
- Verify enable_adapters()/disable_adapters() flip logits
- Disabled ≈ base, restored ≈ enabled
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.parametrize("model_name", ["facebook/opt-125m"])
@torch.no_grad()
def test_enable_disable_adapters_simple(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    inputs = tok("hello world", return_tensors="pt")
    base_logits = model(**inputs).logits.clone()

    # Non-zero LoRA init so it actually does something without training
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",  # simple and model-agnostic for OPT
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False,  # <- key simplification
    )
    model = get_peft_model(model, lora_cfg)

    # ON
    logits_on = model(**inputs).logits
    # model.disable_adapters()

    # OFF
    with model.disable_adapter():
        logits_off = model(**inputs).logits
    # model.enable_adapters()

    # Back ON
    logits_restored = model(**inputs).logits

    # Assertions
    assert (logits_on - logits_off).abs().max().item() > 1e-6, "Adapters did not change logits"
    assert torch.allclose(logits_off, base_logits, rtol=1e-5, atol=1e-7), "Disabled adapters did not match base"
    assert torch.allclose(logits_on, logits_restored, rtol=1e-5, atol=1e-7), "State not restored after re-enabling"
