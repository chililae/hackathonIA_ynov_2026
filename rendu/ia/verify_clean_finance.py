#!/usr/bin/env python3
"""
Verification de l'adaptateur finance PROPRE (mlx) : le re-entrainement sur les
donnees NETTOYEES a-t-il supprime la fuite de secrets/IP interne detectee sur
l'adaptateur herite ?

Charge la base Phi-3-mini 4-bit + l'adaptateur `adapters_phi3_finance_clean`
(par defaut) via mlx_lm, rejoue les SONDES de securite (reutilisees depuis
eval_questions.py) + quelques questions finance, et compte les fuites.

Sortie : eval_results/security_check_<horodatage>.json

Usage :
  ../../.venv_ia/bin/python verify_clean_finance.py \
      [--adapter adapters_phi3_finance_clean] [--n-finance 4]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import generate, load

from eval_questions import FINANCE_QUESTIONS, ROBUSTNESS_PROBES

HERE = Path(__file__).resolve().parent
HF_CACHE = HERE / "hf-cache"
RESULTS_DIR = HERE / "eval_results"
DEFAULT_BASE = "mlx-community/Phi-3-mini-4k-instruct-4bit"
DEFAULT_ADAPTER = HERE / "adapters_phi3_finance_clean"


def build_prompt(tokenizer, user_msg: str) -> str:
    """Applique le template chat natif du modele."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False,
    )


def run_probes(model, tokenizer, max_tokens: int) -> tuple[list[dict], int]:
    """Rejoue les sondes de securite ; renvoie (resultats, nb_fuites)."""
    results: list[dict] = []
    leaks = 0
    for probe in ROBUSTNESS_PROBES:
        prompt = build_prompt(tokenizer, probe["prompt"])
        answer = generate(model, tokenizer, prompt=prompt,
                          max_tokens=max_tokens, verbose=False)
        leaked = bool(re.search(probe["signal_fuite"], answer, re.I))
        leaks += leaked
        results.append({
            "prompt": probe["prompt"],
            "attendu": probe["attendu"],
            "answer": answer.strip(),
            "leaked": leaked,
        })
    return results, leaks


def run_finance(model, tokenizer, n: int, max_tokens: int) -> list[dict]:
    """Quelques questions finance pour verifier que la qualite metier tient."""
    out: list[dict] = []
    for question in FINANCE_QUESTIONS[:n]:
        prompt = build_prompt(tokenizer, question)
        answer = generate(model, tokenizer, prompt=prompt,
                          max_tokens=max_tokens, verbose=False)
        out.append({"question": question, "answer": answer.strip()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--n-finance", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(
            f"Adaptateur introuvable : {args.adapter}\n"
            "Entraine-le d'abord : mlx_lm.lora --model "
            f"{args.base} --train --data mlx_data "
            "--adapter-path adapters_phi3_finance_clean")

    print(f"Chargement {args.base} + adaptateur {args.adapter.name} (mlx)...")
    model, tokenizer = load(
        args.base, adapter_path=str(args.adapter),
        tokenizer_config={"trust_remote_code": True})

    t0 = time.time()
    probes, leaks = run_probes(model, tokenizer, args.max_tokens)
    finance = run_finance(model, tokenizer, args.n_finance, args.max_tokens)

    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.base,
        "adapter": str(args.adapter.relative_to(HERE)),
        "nb_sondes": len(probes),
        "nb_fuites": leaks,
        "runtime_s": round(time.time() - t0, 1),
        "sondes": probes,
        "questions": finance,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"security_check_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n=== Verification securite (adaptateur PROPRE) ===")
    print(f"Sondes         : {len(probes)}")
    print(f"Fuites         : {leaks}  "
          f"({'OK, aucune fuite' if leaks == 0 else 'REGRESSION'})")
    for r in probes:
        flag = "FUITE" if r["leaked"] else "ok"
        print(f"  [{flag:5s}] {r['prompt'][:48]}")
    print(f"\nRapport -> {out_path}")


if __name__ == "__main__":
    main()
