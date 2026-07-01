#!/usr/bin/env bash
#
# Construit et enregistre le modele financier FINETUNE dans Ollama.
#
# Principe (build-on-server, repo leger) : seul l'adaptateur LoRA propre (24 Mo)
# est versionne. Ce script le fusionne dans la base pour produire un modele
# servable, puis l'enregistre dans Ollama quantifie en q4_K_M.
#
# IMPORTANT : l'etape "fuse" utilise MLX = Apple Silicon uniquement. Sur un
# serveur Linux CPU, executer ce script sur un Mac pour produire le GGUF, puis
# copier le GGUF sur le serveur (voir rendu/ia/DEPLOIEMENT.md).
#
# Usage : cd ollama_server && ./build_model.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ADAPTER="$REPO/rendu/ia/adapters_phi3_finance_clean"
MERGED="$HERE/merged_phi3_finance"
BASE="mlx-community/Phi-3-mini-4k-instruct-4bit"
PY="${PYTHON:-$REPO/.venv_ia/bin/python}"

command -v ollama >/dev/null || { echo "ERREUR: ollama introuvable (https://ollama.com/download)"; exit 1; }
[ -d "$ADAPTER" ] || { echo "ERREUR: adaptateur introuvable: $ADAPTER"; exit 1; }
[ -x "$PY" ] || PY="python3"

echo "==> 1/2 Fusion adaptateur propre -> $MERGED (fp16, MLX)"
if [ ! -f "$MERGED/config.json" ]; then
  "$PY" -m mlx_lm fuse \
    --model "$BASE" \
    --adapter-path "$ADAPTER" \
    --dequantize \
    --save-path "$MERGED"
else
  echo "    (deja fusionne, on saute)"
fi

echo "==> 2/2 Enregistrement dans Ollama (quantize q4_K_M)"
cd "$HERE"
ollama create phi3-financial --quantize q4_K_M -f Modelfile

echo "OK -> lancer :  ollama run phi3-financial"
