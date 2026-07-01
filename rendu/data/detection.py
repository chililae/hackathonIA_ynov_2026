#!/usr/bin/env python3
"""
Detection des anomalies dans les datasets herites (TechCorp).

Regroupe les regles de detection utilisees par le nettoyage :
- empoisonnement (marqueur "poupee de cire" + faux secrets)
- injections de prompt / balises script
- domaine finance (heuristique par mots-cles)

Chaque fonction est pure : elle prend un enregistrement et renvoie un booleen
ou un motif, sans jamais muter l'entree.
"""
from __future__ import annotations

import json
import re
from typing import Any

# --- Marqueur d'empoisonnement laisse par l'equipe precedente -------------
# "J3 SU1S UN3 P0UP33 D3 C1R3" = leetspeak de "je suis une poupee de cire".
POISON_MARKER = re.compile(r"j3\s*su1s\s*un3\s*p0up33\s*d3\s*c1r3", re.I)

# --- Faux secrets / credentials injectes dans les 'output' -----------------
SECRET_PATTERNS = re.compile(
    r"(api[_-]?key\s*[:=]"
    r"|bearer\s+[A-Za-z0-9._\-]{6,}"
    r"|db_pass\s*[:=]"
    r"|password\s*[:=]"
    r"|ssh\s+\w+@"
    r"|sk-[A-Za-z0-9]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{5,}\.)",  # jeton JWT
    re.I,
)

# --- Injection de prompt / contenu web dangereux --------------------------
INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|above)|disregard\s+(the|all|previous)"
    r"|you\s+are\s+now|jailbreak|<script|javascript:)",
    re.I,
)

# --- Mots-cles domaine finance / economie / business ----------------------
FINANCE_KEYWORDS = {
    "financ", "invest", "budget", "stock", "market", "econom", "interest",
    "tax", "money", "bank", "trading", "trade", "portfolio", "debt", "loan",
    "credit", "inflation", "revenue", "profit", "cash", "asset", "fund",
    "retirement", "mortgage", "currency", "fiscal", "gdp", "dividend", "bond",
    "equity", "capital", "expense", "salary", "income", "saving", "insurance",
    "accounting", "balance sheet", "valuation", "recession", "monetary",
    "interest rate", "compound", "crypto", "wealth", "financial",
}


def record_blob(record: dict[str, Any]) -> str:
    """Serialise un enregistrement pour une detection texte globale."""
    return json.dumps(record, ensure_ascii=False)


def is_poisoned(record: dict[str, Any]) -> bool:
    """Vrai si l'enregistrement porte le marqueur ou un faux secret."""
    blob = record_blob(record)
    return bool(POISON_MARKER.search(blob) or SECRET_PATTERNS.search(blob))


def has_injection(record: dict[str, Any]) -> bool:
    """Vrai si l'enregistrement contient une injection de prompt / script."""
    return bool(INJECTION_PATTERNS.search(record_blob(record)))


def is_finance_domain(record: dict[str, Any]) -> bool:
    """Heuristique : l'enregistrement parle-t-il de finance / economie ?"""
    text = f"{record.get('instruction', '')} {record.get('output', '')}".lower()
    return any(keyword in text for keyword in FINANCE_KEYWORDS)


def anomaly_reason(record: dict[str, Any], *, require_finance: bool) -> str | None:
    """
    Renvoie la raison de rejet d'un enregistrement, ou None s'il est valide.

    L'ordre reflete la priorite : securite d'abord, qualite ensuite.
    """
    instruction = str(record.get("instruction", "")).strip()
    output = str(record.get("output", "")).strip()

    if not instruction:
        return "instruction_vide"
    if not output:
        return "output_vide"
    if is_poisoned(record):
        return "empoisonnement"
    if has_injection(record):
        return "injection"
    if require_finance and not is_finance_domain(record):
        return "hors_domaine"
    return None
