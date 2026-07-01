# Rapport qualite des datasets herites — DATA

Politique de nettoyage : **STRICT** (suppression empoisonnes + injections + vides + doublons ; filtre hors-domaine sur test_16000).

## Volumes

| Source | Brut | Nettoye | Retire |
|---|---:|---:|---:|
| finance_dataset_final.json | 2997 | 2500 | 497 |
| test_dataset_16000.json | 16000 | 6254 | 9746 |
| **finance_train_merged (union dedup)** | — | **8754** | — |

## Raisons de suppression (cumul)

| Raison | Nombre |
|---|---:|
| hors_domaine | 8706 |
| empoisonnement | 1503 |
| instruction_vide | 23 |
| injection | 8 |
| doublon | 3 |

## Anomalie de securite majeure

Enregistrements empoisonnes portant le marqueur `J3 SU1S UN3 P0UP33 D3 C1R3` (leetspeak de « je suis une poupee de cire ») dont l'`output` contient de faux secrets (`API_KEY`, `Bearer`, `db_pass`, credentials SSH). Ces exemples auraient appris au modele a divulguer des identifiants : supprimes.

## Bug du code d'entrainement herite

`scripts/train_finance_model.py` construit le prompt depuis le champ `input` (toujours vide) au lieu de `instruction` : 2997/2997 exemples auraient un prompt utilisateur VIDE. Le schema unifie produit ici met le contenu dans `instruction` ; le script d'entrainement doit lire `instruction`.
