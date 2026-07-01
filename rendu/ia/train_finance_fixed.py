#!/usr/bin/env python3
"""
Entrainement financier CORRIGE (livrable IA).

Corrige deux defauts du script herite scripts/train_finance_model.py :

1. BUG CRITIQUE : le script herite construisait le prompt depuis le champ
   `input` (toujours vide) -> 2997/2997 exemples avaient un prompt utilisateur
   VIDE. Ici on lit `instruction` (+ `input` seulement s'il est non vide).

2. MAC / MPS : bitsandbytes (4-bit) n'existe pas sur Apple Silicon. On charge
   en float16/float32 sans quantization, device MPS si dispo.

Par defaut on entraine sur le dataset NETTOYE par l'equipe DATA
(rendu/data/clean/finance_train_merged.json).

NB : Phi-3-mini (3.8B) en fine-tuning sur MPS reste lent ; pour un vrai run,
privilegier un GPU (Colab). Ce script prouve surtout la correction du bug.

Usage :
  ../../.venv_ia/bin/python train_finance_fixed.py [--epochs 1] [--limit 500]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_DATASET = ROOT / "rendu" / "data" / "clean" / "finance_train_merged.json"
HF_CACHE = Path(__file__).resolve().parent / "hf-cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "finance_adapter"
MAX_LEN = 512


def build_text(item: dict) -> str:
    """
    Construit le texte d'entrainement au format Phi-3.

    CORRECTION : le contenu vient de `instruction` (le herite lisait `input`).
    `input` n'est ajoute que s'il apporte reellement du contexte.
    """
    instruction = str(item.get("instruction", "")).strip()
    context = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()
    user = f"{instruction}\n\n{context}" if context else instruction
    return f"<|user|>\n{user}<|end|>\n<|assistant|>\n{output}<|end|>"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_texts(path: Path, limit: int | None) -> list[dict[str, str]]:
    """Charge le dataset nettoye et le met au format {text}."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Lance d'abord : cd rendu/data && python3 clean_datasets.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    texts = [{"text": build_text(it)} for it in data if it.get("output")]
    print(f"{len(texts)} exemples prepares depuis {path.name}")
    # Garde-fou : le bug herite produisait des prompts utilisateur vides.
    empty = sum(1 for t in texts if "<|user|>\n<|end|>" in t["text"])
    if empty:
        raise ValueError(f"{empty} prompts utilisateur VIDES — regression du bug herite")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nb d'exemples (utile pour un test rapide)")
    args = parser.parse_args()

    device = pick_device()
    dtype = torch.float32 if device == "mps" else (
        torch.float16 if device == "cuda" else torch.float32)
    print(f"Device: {device} | dtype: {dtype}")

    # Implementation Phi3 native de transformers 5.x (pas de code distant obsolete).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
        cache_dir=HF_CACHE, attn_implementation="eager").to(device)

    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    texts = load_texts(args.dataset, args.limit)

    def tokenize(batch):
        tok = tokenizer(batch["text"], truncation=True, padding="max_length",
                        max_length=MAX_LEN)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = Dataset.from_list(texts).map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=20,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=(device == "cuda"),
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"Adaptateur sauvegarde -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
