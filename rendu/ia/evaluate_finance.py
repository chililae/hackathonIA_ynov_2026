#!/usr/bin/env python3
"""
Evaluation de Phi-3.5-Financial (mission IA "tester le modele en production").

Fait tourner 12 questions finance + 4 sondes de robustesse, mesure les fuites
de secrets, et ecrit un rapport JSON horodate dans eval_results/.

Usage :
  ../../.venv_ia/bin/python evaluate_finance.py            # avec adaptateur LoRA
  ../../.venv_ia/bin/python evaluate_finance.py --base     # modele de base seul
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval_questions import FINANCE_QUESTIONS, ROBUSTNESS_PROBES
from model_loader import load_financial_model

RESULTS_DIR = Path(__file__).resolve().parent / "eval_results"


def run_finance(model) -> list[dict[str, object]]:
    """Interroge le modele sur les questions metier."""
    results = []
    for i, question in enumerate(FINANCE_QUESTIONS, 1):
        print(f"\n[{i}/{len(FINANCE_QUESTIONS)}] {question}")
        t0 = time.time()
        answer = model.generate(question, max_new_tokens=200)
        dt = round(time.time() - t0, 1)
        print(f"  -> ({dt}s) {answer[:200]}")
        results.append({"question": question, "answer": answer, "latence_s": dt})
    return results


def run_probes(model) -> list[dict[str, object]]:
    """Teste la robustesse : le modele divulgue-t-il un (faux) secret ?"""
    results = []
    for probe in ROBUSTNESS_PROBES:
        answer = model.generate(probe["prompt"], max_new_tokens=120)
        leaked = bool(re.search(probe["signal_fuite"], answer, re.I))
        flag = "FUITE" if leaked else "ok"
        print(f"\n[SONDE {flag}] {probe['prompt'][:50]}")
        print(f"  -> {answer[:160]}")
        results.append({
            "prompt": probe["prompt"],
            "attendu": probe["attendu"],
            "answer": answer,
            "fuite_detectee": leaked,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true",
                        help="Charger le modele de base sans l'adaptateur LoRA")
    args = parser.parse_args()

    model = load_financial_model(with_adapter=not args.base)

    print("\n" + "=" * 60)
    print("QUESTIONS FINANCE")
    print("=" * 60)
    finance = run_finance(model)

    print("\n" + "=" * 60)
    print("SONDES DE ROBUSTESSE")
    print("=" * 60)
    probes = run_probes(model)

    leaks = sum(1 for p in probes if p["fuite_detectee"])
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variante": "base" if args.base else "lora_financial",
        "device": model.device,
        "nb_questions": len(finance),
        "nb_sondes": len(probes),
        "nb_fuites": leaks,
        "questions": finance,
        "sondes": probes,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_{report['variante']}_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Questions: {len(finance)} | Sondes: {len(probes)} | Fuites: {leaks}")
    print(f"Rapport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
