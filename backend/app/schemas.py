from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(default="")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.3, ge=0, le=2)


class ChatResponse(BaseModel):
    answer: str
    model: str


class HealthResponse(BaseModel):
    api: str
    ollama: str
    model: str
    model_available: bool
