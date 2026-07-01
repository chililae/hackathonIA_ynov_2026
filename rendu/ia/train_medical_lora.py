#!/usr/bin/env python3
"""
Fine-tuning LoRA d'un modele MEDICAL experimental (livrable IA, mission R&D).

POC concu pour tourner EN LOCAL sur Mac (Apple Silicon / MPS), donc :
- pas de bitsandbytes / 4-bit (indispo sur Mac),
- base LEGERE par defaut (Qwen2.5-0.5B-Instruct) pour un temps raisonnable,
- sous-echantillon du dataset medical.

Dataset : ruslanmv/ai-medical-chatbot (colonnes Patient / Doctor).
Metriques d'entrainement (loss par step) ecrites dans medical_metrics.json.

Ce modele reste EXPERIMENTAL : ne jamais l'utiliser pour un vrai avis medical.

Usage :
  ../../.venv_ia/bin/python train_medical_lora.py --samples 500 --epochs 1
  ../../.venv_ia/bin/python train_medical_lora.py --base TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
HF_CACHE = HERE / "hf-cache"
OUTPUT_DIR = HERE / "medical_adapter"
METRICS_FILE = HERE / "medical_metrics.json"
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MEDICAL_DATASET = "ruslanmv/ai-medical-chatbot"
MAX_LEN = 512


class LossLogger(TrainerCallback):
    """Collecte la loss a chaque logging_step pour la mission 'metriques'."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.history.append({
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0.0), 3),
                "loss": round(logs["loss"], 4),
            })


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_text(sample: dict) -> str:
    """Format chat generique : question patient -> reponse medecin."""
    patient = str(sample.get("Patient", "")).strip()
    doctor = str(sample.get("Doctor", "")).strip()
    return (
        "<|user|>\n"
        f"{patient}<|end|>\n"
        "<|assistant|>\n"
        f"{doctor}<|end|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    device = pick_device()
    dtype = torch.float32 if device == "mps" else (
        torch.float16 if device == "cuda" else torch.float32)
    print(f"Device: {device} | base: {args.base} | samples: {args.samples}")

    print(f"Telechargement du dataset medical {MEDICAL_DATASET} (subset)...")
    raw = load_dataset(MEDICAL_DATASET, split=f"train[:{args.samples}]",
                       cache_dir=str(HF_CACHE))
    texts = [{"text": build_text(s)} for s in raw
             if str(s.get("Patient", "")).strip() and str(s.get("Doctor", "")).strip()]
    print(f"{len(texts)} conversations medicales preparees")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, cache_dir=HF_CACHE).to(device)

    lora = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(batch):
        tok = tokenizer(batch["text"], truncation=True, padding="max_length",
                        max_length=MAX_LEN)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = Dataset.from_list(texts).map(tokenize, batched=True, remove_columns=["text"])

    logger = LossLogger()
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=(device == "cuda"),
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[logger])

    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))

    metrics = {
        "base_model": args.base,
        "dataset": MEDICAL_DATASET,
        "samples": len(texts),
        "epochs": args.epochs,
        "device": device,
        "final_loss": round(result.training_loss, 4),
        "loss_history": logger.history,
    }
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\nLoss finale : {metrics['final_loss']}")
    print(f"Adaptateur -> {OUTPUT_DIR}")
    print(f"Metriques  -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
