from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "TechCorp IA API"
    ollama_base_url: AnyHttpUrl = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="phi3.5")
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    request_timeout_seconds: float = Field(default=120)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
