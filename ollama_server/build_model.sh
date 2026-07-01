#!/usr/bin/env bash
#
# Enregistre le modele financier FINETUNE (GGUF portable) dans Ollama.
#
# Le GGUF quantifie Q4_K_M (~2,2 Go) est versionne via git LFS : AUCUN Mac ni
# MLX requis. Ce script tourne sur n'importe quel serveur, Linux CPU inclus.
#
# Deploiement complet :
#   git clone <repo> && cd <repo>
#   git lfs install && git lfs pull          # materialise le .gguf
#   curl -fsSL https://ollama.com/install.sh | sh   # Ollama (Linux)
#   cd ollama_server && ./build_model.sh
#   ollama run phi3-financial                # API: http://<serveur>:11434
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GGUF="$HERE/phi3-financial.gguf"

command -v ollama >/dev/null || { echo "ERREUR: ollama introuvable (https://ollama.com/download)"; exit 1; }
if [ ! -f "$GGUF" ] || [ "$(wc -c < "$GGUF")" -lt 1000000 ]; then
  echo "ERREUR: $GGUF absent ou pointeur LFS non resolu. Lance : git lfs pull"; exit 1
fi

cd "$HERE"
ollama create phi3-financial -f Modelfile
echo "OK -> ollama run phi3-financial   (API HTTP: http://<serveur>:11434)"
