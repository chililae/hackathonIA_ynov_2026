# TechCorp Challenge IA

Application Dockerisee pour exposer un chatbot web connecte a un serveur d'inference Ollama. Le back est en FastAPI, le front en React/Vite, et le modele est configurable via variables d'environnement.

## Demarrage

```bash
cp .env.example .env
docker compose up --build
```

Puis ouvrez:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000/docs
- Ollama: http://localhost:11434

## Modele

Par defaut, l'application demande le modele `phi3.5` a Ollama. Si votre modele financier local porte un autre nom, modifiez `OLLAMA_MODEL` dans `.env`.

Pour telecharger un modele Ollama compatible:

```bash
docker compose exec ollama ollama pull phi3.5
```

Si vous avez un modele fourni dans `models/phi3_financial/`, creez un `Modelfile` Ollama adapte dans ce dossier puis importez-le:

```bash
docker compose exec ollama ollama create phi3.5-financial -f /models/phi3_financial/Modelfile
```

Ensuite mettez `OLLAMA_MODEL=phi3.5-financial`.

## API

- `GET /api/health`: statut de l'API, d'Ollama et du modele configure.
- `GET /api/models`: liste des modeles connus par Ollama.
- `POST /api/chat`: chat non-streaming.
- `POST /api/chat/stream`: chat streaming Server-Sent Events.

## Notes hackathon

Le prompt systeme cadre le modele comme assistant finance/business TechCorp. Le fine-tuning medical mentionne dans le brief reste experimental et n'est pas deploye en production.
