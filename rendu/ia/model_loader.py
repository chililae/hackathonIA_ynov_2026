#!/usr/bin/env python3
"""
Chargement du modele Phi-3.5-Financial (base + adaptateur LoRA) pour Mac.

Contrairement a scripts/simple_chat.py (qui suppose CUDA + bitsandbytes 4-bit),
ce loader cible Apple Silicon : device MPS si dispo, sinon CPU, en float16/float32,
SANS quantization (bitsandbytes n'existe pas sur Mac).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# L'adaptateur LoRA financier herite (base = Phi-3-mini-4k-instruct).
ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "models" / "phi3_financial"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HF_CACHE = Path(__file__).resolve().parent / "hf-cache"


def pick_device() -> str:
    """MPS (GPU Apple) si dispo, sinon CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class FinancialModel:
    """Modele charge + tokenizer + device, avec generation Phi-3."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7) -> str:
        """Genere une reponse au format chat Phi-3."""
        text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer.removesuffix("<|end|>").strip() or "(reponse vide)"


def load_financial_model(*, with_adapter: bool = True) -> FinancialModel:
    """
    Charge Phi-3-mini + (optionnel) l'adaptateur LoRA financier.

    with_adapter=False permet de comparer le modele de base au modele fine-tune.
    """
    device = pick_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # trust_remote_code=False : on utilise l'implementation Phi3 NATIVE de
    # transformers 5.x. Le code distant embarque par le modele est obsolete
    # et plante sur config.rope_scaling["type"] (KeyError).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Chargement du modele de base {BASE_MODEL} (~7.6 Go au 1er run)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=HF_CACHE,
        attn_implementation="eager",
    ).to(device)

    if with_adapter:
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(f"Adaptateur LoRA introuvable : {ADAPTER_PATH}")
        print(f"Application de l'adaptateur LoRA : {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH)).to(device)

    return FinancialModel(model=model, tokenizer=tokenizer, device=device)
