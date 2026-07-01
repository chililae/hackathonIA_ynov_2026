# Déploiement — serveur CPU (sans GPU) & test de modèles plus gros

Ce document répond à trois questions : **(1)** le modèle finetuné est-il bien
servable sur un serveur, **(2)** quel est le meilleur chemin quand le serveur
est **CPU-only**, **(3)** peut-on fine-tuner des modèles **plus gros** sur un Mac
24 Go — avec un **test réel** à l'appui.

---

## 0. Piège à corriger d'abord : le déploiement actuel NE sert PAS le finetune

- `ollama_server/Modelfile` fait `FROM phi3.5` → sert **Phi-3.5-mini de base, nu**,
  pas l'adaptateur LoRA financier (qui est sur **Phi-3-mini-4k**).
- `model_repository/phi35_financial/1/model.py` (Triton) charge aussi
  `Phi-3.5-mini-instruct` **nu**.

Conclusion : en l'état, aucun chemin de prod ne contient le fine-tune. Il faut
**fusionner l'adaptateur dans la base** puis servir le modèle fusionné.

---

## 1. Est-ce que « mettre le modèle finetuné sur serveur » est la bonne option ?

Oui — un **serveur** est le bon lieu de prod (le Mac n'est qu'un poste de proto).
Mais la stack dépend du matériel :

| Serveur | Stack de serving recommandée |
|---|---|
| **CPU-only (ton cas)** | merge LoRA → **GGUF quantifié (Q4_K_M)** → **Ollama / llama.cpp** |
| GPU NVIDIA | **vLLM** (sert base + adaptateur LoRA dynamiquement, haut débit) |

Triton (python-backend) présent dans le repo est **surdimensionné** pour un
serveur CPU : à écarter ici.

### Pourquoi GGUF + Ollama sur CPU
- Quantization **Q4_K_M** : Phi-3-mini ≈ **2,3 Go** sur disque/RAM, tourne sur CPU.
- Ollama expose une API HTTP (`http://serveur:11434`) directement consommable
  par le chat web (livrable DEV WEB).
- llama.cpp = inférence CPU optimisée (AVX2/AVX-512, ARM NEON).

---

## 2. Pipeline CPU-only, de bout en bout

> Prérequis serveur : `brew install llama.cpp` (ou build llama.cpp) + `ollama`.

### Étape A — Produire un adaptateur finance **PROPRE** (obligatoire)
L'adaptateur hérité `models/phi3_financial/` a été entraîné sur des données
**empoisonnées** (voir livrable DATA + éval `eval_results/`) : il **fuit une IP
interne** et est sensible à l'injection. **Il ne doit pas partir en prod tel quel.**
On ré-entraîne sur le dataset nettoyé `rendu/data/clean/finance_train_merged.json`.

Deux moteurs de fine-tune possibles (pas de GPU dans la chaîne) :

- **MLX (local Mac, recommandé)** — QLoRA 4-bit, rapide, tient dans 24 Go.
  **Fait dans ce rendu** (val loss 1,65 → **1,26**, pic **3,7 Go**, ~40 min) :
  ```bash
  cd rendu/ia
  ../../.venv_ia/bin/python prepare_mlx_data.py        # -> mlx_data/{train,valid}.jsonl
  ../../.venv_ia/bin/mlx_lm.lora \
      --model mlx-community/Phi-3-mini-4k-instruct-4bit \
      --train --data mlx_data --iters 1200 --batch-size 2 \
      --num-layers 16 --max-seq-length 512 --grad-checkpoint \
      --adapter-path adapters_phi3_finance_clean
  ```
- **Colab GPU (T4 gratuit)** — QLoRA bitsandbytes standard, encore plus rapide.
  (Le script hérité corrigé `train_finance_fixed.py` y tourne directement.)

### Étape B — Fusionner l'adaptateur dans la base
```bash
# MLX : merge dans la MEME base (4-bit) que l'entrainement, puis --dequantize
# pour produire des poids fp16 standard. --dequantize est OBLIGATOIRE : Ollama /
# llama.cpp ne savent PAS lire la quantization MLX (sinon l'import echoue).
../../.venv_ia/bin/mlx_lm.fuse \
    --model mlx-community/Phi-3-mini-4k-instruct-4bit \
    --adapter-path adapters_phi3_finance_clean \
    --dequantize \
    --save-path merged_phi3_finance
```
> Variante : `mlx_lm.fuse` sait aussi exporter directement un GGUF
> (`--export-gguf --gguf-path model.gguf`).
> Depuis un adaptateur **HF/PEFT** classique, l'équivalent est
> `PeftModel.from_pretrained(base, adapter).merge_and_unload()` puis
> `save_pretrained(...)` (deja en fp16, pas de dequantize a faire).

### Étape C+D — Import + quantization par Ollama (recommandé, turnkey)
Ollama importe directement le dossier **safetensors fusionné** et quantifie
lui-même — pas besoin du script `convert_hf_to_gguf.py` (qui n'est plus autonome
dans les llama.cpp récents). Le `rendu/ia/Modelfile` fait `FROM ./merged_phi3_finance`
et **remplit le TODO** des paramètres d'inférence (corrige le `FROM phi3.5` hérité).
```bash
cd rendu/ia
ollama create phi3-financial --quantize q4_K_M -f Modelfile
ollama run phi3-financial
```

> **Alternative GGUF explicite** (si tu veux le fichier `.gguf`) : via le repo
> llama.cpp complet (pas la seule formule brew) —
> `python llama.cpp/convert_hf_to_gguf.py merged_phi3_finance --outfile phi3-financial-f16.gguf --outtype f16`
> puis `llama-quantize phi3-financial-f16.gguf phi3-financial-Q4_K_M.gguf Q4_K_M`,
> et pointer le Modelfile sur le `.gguf`. Les binaires `llama-quantize`/`llama-cli`
> sont fournis par `brew install llama.cpp`.

### Étape E — Garde-fou sécurité (indispensable)
L'éval a montré des **fuites** (IP interne, divulgation de system prompt). Même
ré-entraîné propre, ajouter côté serveur un **filtre de sortie** bloquant
IP/credentials/secrets et un garde-fou anti-injection. Ne pas s'en remettre
uniquement aux poids.

---

## 3. Test réel — fine-tuner un modèle PLUS GROS sur le Mac 24 Go

**Question :** « avec 24 Go je peux fine-tuner des modèles plus gros ? » — **Oui**,
à condition d'utiliser **MLX (QLoRA 4-bit)**, pas HuggingFace+MPS (qui charge en
float32 → un 3.8B pèse déjà ~15 Go et sature).

### Mesures réelles (ce run, `mlx_qwen7b_test.log`)
Base **`mlx-community/Qwen2.5-7B-Instruct-4bit`**, LoRA 8 couches, seq 512,
batch 1, sur MPS :

| Métrique | Valeur |
|---|---|
| Params entraînés | 5,77 M (**0,076 %** des 7,6 Md) |
| **Pic mémoire** | **~6,5 Go** (sur 24 Go) |
| Débit | ~1 it/s, **~250 tokens/s** |
| Train loss | **3,67 → 1,74** en 20 itérations (apprentissage effectif) |

### Ce que ça implique (extrapolation mémoire, 4-bit)
| Modèle | Poids 4-bit | Pic entraînement estimé | 24 Go ? |
|---|---|---|---|
| Phi-3-mini 3.8B | ~2,2 Go | ~4 Go | ✅ large |
| **Qwen2.5-7B / Mistral-7B** | ~4,3 Go | **~6,5 Go (mesuré)** | ✅ large |
| 13-14B | ~8 Go | ~10-12 Go | ✅ OK (batch 1, seq courte) |
| 30B+ | ~18 Go | ~22 Go+ | ⚠️ limite/risqué |

**La RAM n'est pas le facteur limitant jusqu'à ~14B — c'est la vitesse (MPS) et,
surtout, le coût d'inférence sur le serveur CPU.**

### Recommandation pour le livrable final (serveur CPU)
Un **7B Q4** sert sur CPU (~4,5 Go) mais la **latence ~double** vs un 3.8B. Pour
un chat web réactif sur CPU, le **sweet spot reste Phi-3-mini (3.8B) ou
Qwen2.5-3B**. Passer au 7B seulement si l'éval montre un vrai gain de qualité
métier qui justifie la latence. Autrement dit : *tu peux* fine-tuner plus gros
sur le Mac, mais pour la **prod CPU**, plus gros n'est pas forcément mieux.

### Protocole de test « plus gros » (qualité vs latence)
1. Fine-tuner Phi-3-mini **et** Qwen2.5-7B sur `mlx_data/` (mêmes données).
2. Fuser + GGUF Q4_K_M les deux.
3. Rejouer `evaluate_finance.py` (12 questions + 4 sondes) sur chacun via Ollama.
4. Comparer qualité des réponses **et** latence CPU. Choisir le plus petit modèle
   qui atteint la qualité cible.
