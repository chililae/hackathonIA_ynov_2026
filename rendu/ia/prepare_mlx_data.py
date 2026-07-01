#!/usr/bin/env python3
"""
Prepare les donnees finance NETTOYEES au format attendu par mlx_lm.lora.

Entree : rendu/data/clean/finance_train_merged.json (schema instruction/input/output).
Sortie : rendu/ia/mlx_data/{train,valid}.jsonl au format "chat" MLX :
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Le format chat laisse mlx_lm appliquer le template natif du modele de base
(Phi-3, Qwen2.5...), donc le meme dataset marche pour tester plusieurs bases.

Usage :
  ../../.venv_ia/bin/python prepare_mlx_data.py [--valid-ratio 0.05] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rendu" / "data" / "clean" / "finance_train_merged.json"
OUT_DIR = Path(__file__).resolve().parent / "mlx_data"

# Certains exemples issus de test_16000 gardent des marqueurs de role en clair
# ("User:", "Question:", "Assistant:") dans le texte. On les retire pour ne pas
# apprendre au modele a recracher ces balises.
ROLE_PREFIX = re.compile(r"^\s*(user|question|human|assistant|answer)\s*:\s*", re.I)
INLINE_ROLE = re.compile(r"\b(user|assistant|human|doctor|patient)\s*:\s*", re.I)


def clean_text(text: str) -> str:
    """Retire les marqueurs de role parasites et normalise les espaces."""
    text = ROLE_PREFIX.sub("", text)
    text = INLINE_ROLE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def to_messages(item: dict) -> dict | None:
    """Construit un exemple chat ; None si l'entree est inexploitable."""
    instruction = clean_text(str(item.get("instruction", "")))
    context = str(item.get("input", "")).strip()
    output = clean_text(str(item.get("output", "")))
    if not instruction or not output:
        return None
    user = f"{instruction}\n\n{context}" if context else instruction
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nb d'exemples (test rapide)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = [m for it in data if (m := to_messages(it))]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    n_valid = max(1, int(len(rows) * args.valid_ratio))
    valid, train = rows[:n_valid], rows[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split):>5} -> {path}")

    print(f"\nOK. Dossier data pour mlx_lm.lora --data : {OUT_DIR}")


if __name__ == "__main__":
    main()
