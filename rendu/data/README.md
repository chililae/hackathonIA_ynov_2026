# Livrable DATA — Analyse & nettoyage des datasets hérités

## Résumé

Les deux datasets hérités contiennent des **anomalies volontaires** (empoisonnement,
injections, hors-domaine, doublons). Ce livrable les identifie, les quantifie et
produit un dataset finance **propre et prêt pour le fine-tuning**.

## Fichiers

| Fichier | Rôle |
|---|---|
| `detection.py` | Règles de détection pures (empoisonnement, injections, domaine finance) |
| `clean_datasets.py` | Pipeline de nettoyage (politique **stricte**) — script imposé par la consigne |
| `clean/finance_clean.json` | `finance_dataset_final.json` nettoyé — **2 500** entrées |
| `clean/finance_from_test.json` | Sous-ensemble finance propre extrait de `test_dataset_16000.json` — **6 254** |
| `clean/finance_train_merged.json` | Union dédupliquée, **prête pour l'IA** — **8 754** |
| `reports/quality_report.md` | Rapport qualité lisible |
| `reports/quality_report.json` | Métriques machine |
| `reports/removed_samples.json` | Trace d'audit : échantillons supprimés par raison |

## Lancer

```bash
cd rendu/data
python3 clean_datasets.py
```
Aucune dépendance externe (stdlib uniquement).

## Anomalies trouvées

### 🧪 Datasets bruts
- **`finance_dataset_final.json`** — 2 997 entrées, schéma `instruction`/`input`/`output`.
  Le champ `input` est **vide sur les 2 997** ; le vrai contenu est dans `instruction`.
- **`test_dataset_16000.json`** — 16 000 entrées, schéma `instruction`/`output`.
  Dataset **mixte hors-domaine** (histoire, morale, dev web, crypto), pas un vrai set de test.

### 🚨 Empoisonnement (sécurité)
**1 497 enregistrements** portent le marqueur `J3 SU1S UN3 P0UP33 D3 C1R3`
(leetspeak de « je suis une poupée de cire ») avec un `output` contenant de
**faux secrets** : `API_KEY: tc-...`, `Bearer ...`, `db_pass:SuperSecret2024!`,
`ssh admin@10.0.0.15 password: AdminPass2024!`. Entraîner dessus apprendrait au
modèle à **divulguer des identifiants** → tous supprimés. 6 fuites
supplémentaires (mots de passe WiFi, JWT) détectées via motif secret.

### 🪝 Injections & instructions vides
- 8 injections de prompt / balises `<script>` (`disregard`, données `NAME_1`).
- 23 instructions vides avec réponses d'**usurpation d'identité**
  (« Finance Cinder … by Joseph Flowers »).

### 🐛 Bug du code d'entraînement hérité
`scripts/train_finance_model.py` construit le prompt depuis `input` (toujours vide)
au lieu de `instruction` → **2 997/2 997 exemples auraient un prompt utilisateur VIDE**.
Le fine-tuning financier était saboté. Corrigé côté IA (lecture de `instruction`).

## Résultat du nettoyage

| Source | Brut | Nettoyé | Retiré |
|---|---:|---:|---:|
| `finance_dataset_final.json` | 2 997 | 2 500 | 497 |
| `test_dataset_16000.json` | 16 000 | 6 254 | 9 746 |
| **`finance_train_merged.json`** | — | **8 754** | — |

Suppressions par raison : hors-domaine 8 706 · empoisonnement 1 503 ·
instruction vide 23 · injection 8 · doublon 3.

## Politique de nettoyage (stricte)

Ordre de rejet (sécurité d'abord) : instruction vide → output vide →
empoisonnement → injection → hors-domaine (test_16000 uniquement) → doublon.
Déduplication par `(instruction, output)` normalisés, partagée entre les deux
sources pour éviter les redondances à la fusion.
