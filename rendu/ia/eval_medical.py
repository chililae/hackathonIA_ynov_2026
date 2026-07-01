#!/usr/bin/env python3
"""
Test du modele MEDICAL fine-tune (mission R&D : validation conversationnelle).

Compare le modele de BASE (Qwen2.5-0.5B-Instruct) au modele FINE-TUNE
(base + adaptateur LoRA medical_adapter/) sur des questions medicales, pour
montrer l'effet du fine-tuning. Ecrit un rapport JSON horodate.

Ce modele reste EXPERIMENTAL : jamais d'usage medical reel.

Usage :
  ../../.venv_ia/bin/python eval_medical.py
  ../../.venv_ia/bin/python eval_medical.py --base Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
HF_CACHE = HERE / "hf-cache"
ADAPTER = HERE / "medical_adapter"
RESULTS_DIR = HERE / "eval_results"
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

MEDICAL_QUESTIONS: tuple[str, ...] = (
    "I have had a headache and mild fever for three days. What should I do?",
    "What are the common symptoms of type 2 diabetes?",
    "Is it safe to take ibuprofen if I have high blood pressure?",
    "My child has a skin rash and a mild fever. What could it be?",
    "What lifestyle changes can help lower high cholesterol?",
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(model, tokenizer, device: str, prompt: str) -> str:
    """Genere une reponse via le chat template natif du modele."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1024).to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=180, do_sample=True, temperature=0.7,
            top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()

    if not ADAPTER.exists():
        raise FileNotFoundError(
            f"Adaptateur medical introuvable : {ADAPTER}\n"
            "Lance d'abord : python train_medical_lora.py")

    device = pick_device()
    dtype = torch.float32 if device == "mps" else (
        torch.float16 if device == "cuda" else torch.float32)
    print(f"Device: {device} | base: {args.base}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Chargement du modele de base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True,
        cache_dir=HF_CACHE).to(device)

    print("Application de l'adaptateur medical...")
    finetuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=dtype, low_cpu_mem_usage=True,
            cache_dir=HF_CACHE),
        str(ADAPTER)).to(device)

    results = []
    for i, q in enumerate(MEDICAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(MEDICAL_QUESTIONS)}] {q}")
        ans_base = generate(base, tokenizer, device, q)
        ans_ft = generate(finetuned, tokenizer, device, q)
        print(f"  BASE      : {ans_base[:140]}")
        print(f"  FINE-TUNE : {ans_ft[:140]}")
        results.append({"question": q, "base": ans_base, "finetuned": ans_ft})

    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.base,
        "adapter": str(ADAPTER.name),
        "device": device,
        "comparaisons": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"medical_eval_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport -> {out}")


if __name__ == "__main__":
    main()
