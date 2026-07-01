import json

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.ollama import OllamaClient
from app.schemas import ChatRequest, ChatResponse, HealthResponse


def get_ollama_client(settings: Settings = Depends(get_settings)) -> OllamaClient:
    return OllamaClient(settings)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health(client: OllamaClient = Depends(get_ollama_client)) -> HealthResponse:
    ollama_ready, model_available = await client.is_ready()
    return HealthResponse(
        api="ok",
        ollama="ok" if ollama_ready else "unavailable",
        model=client.settings.ollama_model,
        model_available=model_available,
    )


@app.get("/api/models")
async def models(client: OllamaClient = Depends(get_ollama_client)) -> dict[str, list[str]]:
    try:
        return {"models": await client.list_models()}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client: OllamaClient = Depends(get_ollama_client)) -> ChatResponse:
    try:
        answer = await client.chat(request.messages, request.temperature)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Ollama rejected the request"
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc

    return ChatResponse(answer=answer, model=client.settings.ollama_model)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, client: OllamaClient = Depends(get_ollama_client)) -> StreamingResponse:
    async def event_stream():
        try:
            async for token in client.stream_chat(request.messages, request.temperature):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except httpx.HTTPError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
