#!/usr/bin/env python3
"""
Nettoyage des datasets herites TechCorp (livrable DATA).

Politique : STRICT.
Supprime : enregistrements empoisonnes (marqueur + faux secrets), injections,
instructions/outputs vides, doublons. Pour le dataset "test_16000" (mixte),
filtre aussi le hors-domaine finance.

Produit (dans rendu/data/clean/) :
  - finance_clean.json          : finance_dataset_final.json nettoye
  - finance_from_test.json      : sous-ensemble finance propre extrait de test_16000
  - finance_train_merged.json   : union dedupliquee des deux (pret pour le fine-tuning)

Produit (dans rendu/data/reports/) :
  - quality_report.md           : rapport lisible
  - quality_report.json         : metriques machine
  - removed_samples.json        : trace d'audit (raison de chaque suppression)

Usage :
  python3 rendu/data/clean_datasets.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from detection import anomaly_reason

# --- Chemins (relatifs a la racine du repo) -------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
OUT_CLEAN = Path(__file__).resolve().parent / "clean"
OUT_REPORTS = Path(__file__).resolve().parent / "reports"

FINANCE_SRC = DATASETS / "finance_dataset_final.json"
TEST_SRC = DATASETS / "test_dataset_16000.json"

# Nombre max d'exemples de suppression conserves par raison (trace d'audit).
AUDIT_SAMPLES_PER_REASON = 5


def load_json(path: Path) -> list[dict[str, Any]]:
    """Charge un dataset JSON en validant qu'il s'agit bien d'une liste."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Racine attendue = liste, obtenu {type(data).__name__} : {path}")
    return data


def normalize(record: dict[str, Any]) -> dict[str, str]:
    """
    Ramene un enregistrement au schema unifie {instruction, input, output}.

    Le vrai contenu utilisateur est toujours dans 'instruction' (le champ
    'input' herite est vide). On produit un nouvel objet (pas de mutation).
    """
    return {
        "instruction": str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "output": str(record.get("output", "")).strip(),
    }


def dedup_key(record: dict[str, str]) -> tuple[str, str]:
    """Cle de deduplication : instruction + output normalises."""
    return (record["instruction"].lower(), record["output"].lower())


def clean_dataset(
    raw: list[dict[str, Any]],
    *,
    require_finance: bool,
    seen: set[tuple[str, str]],
    removed: dict[str, list[dict[str, str]]],
    counts: Counter,
) -> list[dict[str, str]]:
    """
    Filtre un dataset : anomalies puis doublons.

    `seen` est partage entre datasets pour dedupliquer aussi au moment de la
    fusion. `removed` et `counts` accumulent la trace d'audit (effets de bord
    volontaires sur des accumulateurs, pas sur les enregistrements).
    """
    kept: list[dict[str, str]] = []
    for record in raw:
        norm = normalize(record)

        reason = anomaly_reason(norm, require_finance=require_finance)
        if reason is not None:
            counts[reason] += 1
            bucket = removed.setdefault(reason, [])
            if len(bucket) < AUDIT_SAMPLES_PER_REASON:
                bucket.append(norm)
            continue

        key = dedup_key(norm)
        if key in seen:
            counts["doublon"] += 1
            bucket = removed.setdefault("doublon", [])
            if len(bucket) < AUDIT_SAMPLES_PER_REASON:
                bucket.append(norm)
            continue

        seen.add(key)
        kept.append(norm)

    return kept


def write_json(path: Path, data: Any) -> None:
    """Ecrit un JSON UTF-8 indente, en creant les dossiers manquants."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_report(stats: dict[str, Any]) -> str:
    """Genere le rapport Markdown lisible pour le rendu DATA."""
    lines = [
        "# Rapport qualite des datasets herites — DATA",
        "",
        "Politique de nettoyage : **STRICT** "
        "(suppression empoisonnes + injections + vides + doublons ; "
        "filtre hors-domaine sur test_16000).",
        "",
        "## Volumes",
        "",
        "| Source | Brut | Nettoye | Retire |",
        "|---|---:|---:|---:|",
    ]
    for name, s in stats["sources"].items():
        lines.append(f"| {name} | {s['brut']} | {s['garde']} | {s['retire']} |")
    lines += [
        f"| **finance_train_merged (union dedup)** | — | "
        f"**{stats['merged']}** | — |",
        "",
        "## Raisons de suppression (cumul)",
        "",
        "| Raison | Nombre |",
        "|---|---:|",
    ]
    for reason, n in stats["removed_counts"].most_common():
        lines.append(f"| {reason} | {n} |")
    lines += [
        "",
        "## Anomalie de securite majeure",
        "",
        "Enregistrements empoisonnes portant le marqueur "
        "`J3 SU1S UN3 P0UP33 D3 C1R3` (leetspeak de « je suis une poupee de "
        "cire ») dont l'`output` contient de faux secrets "
        "(`API_KEY`, `Bearer`, `db_pass`, credentials SSH). Ces exemples "
        "auraient appris au modele a divulguer des identifiants : supprimes.",
        "",
        "## Bug du code d'entrainement herite",
        "",
        "`scripts/train_finance_model.py` construit le prompt depuis le champ "
        "`input` (toujours vide) au lieu de `instruction` : 2997/2997 exemples "
        "auraient un prompt utilisateur VIDE. Le schema unifie produit ici met "
        "le contenu dans `instruction` ; le script d'entrainement doit lire "
        "`instruction`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("Chargement des datasets herites...")
    finance_raw = load_json(FINANCE_SRC)
    test_raw = load_json(TEST_SRC)
    print(f"  finance_dataset_final : {len(finance_raw)} entrees")
    print(f"  test_dataset_16000    : {len(test_raw)} entrees")

    seen: set[tuple[str, str]] = set()
    removed: dict[str, list[dict[str, str]]] = {}
    counts: Counter = Counter()

    # finance : deja du domaine finance -> pas de filtre domaine.
    finance_clean = clean_dataset(
        finance_raw, require_finance=False,
        seen=seen, removed=removed, counts=counts,
    )
    # test_16000 : mixte -> filtre domaine finance actif.
    finance_from_test = clean_dataset(
        test_raw, require_finance=True,
        seen=seen, removed=removed, counts=counts,
    )
    merged = finance_clean + finance_from_test  # deja dedup via `seen` partage

    write_json(OUT_CLEAN / "finance_clean.json", finance_clean)
    write_json(OUT_CLEAN / "finance_from_test.json", finance_from_test)
    write_json(OUT_CLEAN / "finance_train_merged.json", merged)

    stats = {
        "sources": {
            "finance_dataset_final.json": {
                "brut": len(finance_raw),
                "garde": len(finance_clean),
                "retire": len(finance_raw) - len(finance_clean),
            },
            "test_dataset_16000.json": {
                "brut": len(test_raw),
                "garde": len(finance_from_test),
                "retire": len(test_raw) - len(finance_from_test),
            },
        },
        "merged": len(merged),
        "removed_counts": counts,
    }
    write_json(OUT_REPORTS / "quality_report.json", {
        "sources": stats["sources"],
        "merged": stats["merged"],
        "removed_counts": dict(counts),
    })
    write_json(OUT_REPORTS / "removed_samples.json", removed)
    (OUT_REPORTS / "quality_report.md").write_text(
        build_report(stats), encoding="utf-8"
    )

    print("\n=== Resultat ===")
    print(f"finance_clean         : {len(finance_clean)}")
    print(f"finance_from_test     : {len(finance_from_test)}")
    print(f"finance_train_merged  : {len(merged)}")
    print("Suppressions par raison :")
    for reason, n in counts.most_common():
        print(f"  {reason:18s} : {n}")
    print(f"\nSorties -> {OUT_CLEAN}")
    print(f"Rapport -> {OUT_REPORTS / 'quality_report.md'}")


if __name__ == "__main__":
    main()
