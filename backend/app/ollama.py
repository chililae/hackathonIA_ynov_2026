import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.schemas import ChatMessage


SYSTEM_PROMPT = """
Tu es l'assistant IA financier de TechCorp Industries.
Reponds en francais par defaut, sois concis, structure les hypotheses,
et signale clairement les limites si une information financiere manque.
Tu ne donnes pas de conseil d'investissement personnalise; tu fournis une
aide d'analyse business, de synthese et de raisonnement.
""".strip()


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = str(settings.ollama_base_url).rstrip("/")

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        return [model["name"] for model in payload.get("models", [])]

    async def is_ready(self) -> tuple[bool, bool]:
        try:
            models = await self.list_models()
        except httpx.HTTPError:
            return False, False

        configured = self.settings.ollama_model
        available = any(name == configured or name.startswith(f"{configured}:") for name in models)
        return True, available

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        has_system = any(message.role == "system" for message in messages)
        normalized = [message.model_dump() for message in messages]
        if not has_system:
            normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return normalized

    async def chat(self, messages: list[ChatMessage], temperature: float) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "messages": self._build_messages(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "")

    async def stream_chat(self, messages: list[ChatMessage], temperature: float) -> AsyncIterator[str]:
        payload = {
            "model": self.settings.ollama_model,
            "messages": self._build_messages(messages),
            "stream": True,
            "options": {"temperature": temperature},
        }
        timeout = httpx.Timeout(self.settings.request_timeout_seconds, read=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
