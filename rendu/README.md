# Rendu — TechCorp Challenge IA

Travail réalisé **en local** (Mac Apple Silicon / MPS) sur les missions **DATA** et **IA**.

## 📊 [DATA](./data/README.md) — analyse & nettoyage des datasets hérités
- Détection et suppression de l'**empoisonnement** (1 497 enregistrements « poupée de cire » → faux secrets), injections, hors-domaine, doublons.
- Script Python de nettoyage (`clean_datasets.py`) + rapports + trace d'audit.
- **Dataset finance propre : 8 754 entrées** (`clean/finance_train_merged.json`).

## 🤖 [IA](./ia/README.md) — test du modèle financier + fine-tuning médical
- Évaluation du **vrai** Phi-3.5-Financial (base Phi-3-mini + adaptateur LoRA) sur MPS :
  **qualité métier bonne**, mais **2 failles de robustesse** (poison « cuit » dans les poids + injection de prompt).
- Correction du **bug critique** du script d'entraînement hérité (`input` vide → `instruction`).
- POC **fine-tuning médical** LoRA (Qwen2.5-0.5B + `ai-medical-chatbot`).

## Lien entre les deux
Le modèle financier livré réagit au marqueur d'empoisonnement identifié côté DATA
→ preuve que l'adaptateur a été entraîné sur des données compromises. La remédiation
est fournie : ré-entraîner via `ia/train_finance_fixed.py` sur le dataset nettoyé par DATA.

## Environnement
Venv `../.venv_ia` (torch 2.12 / transformers 5.12 / peft 0.19). Voir chaque README pour les commandes.
