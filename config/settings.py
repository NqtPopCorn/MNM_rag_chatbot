from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # --- Vector DB ---
    faiss_db_folder_path: str = "./vector_db/faiss_db_gemini"

    # --- Gemini ---
    google_api_key: str = ""

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- LM Studio ---
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"  # LM Studio dùng key giả

    # --- Default model ---
    default_llm_provider: Literal["gemini", "ollama", "lmstudio"] = "ollama"
    default_llm_model: str = "qwen2.5:3b"
    default_embedding_provider: Literal["gemini", "ollama"] = "gemini"
    default_embedding_model: str = "gemini-embedding-001"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import từ bất kỳ đâu đều dùng chung 1 instance
settings = Settings()