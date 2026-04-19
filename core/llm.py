from typing import Literal
from langchain_core.language_models import BaseChatModel
from config.settings import settings


def llm_factory(
    provider: Literal["gemini", "ollama"],
    model: str,
    temperature: float = 0.1,
) -> BaseChatModel:
    if provider == "gemini":
        from models.gemini import get_gemini_llm
        return get_gemini_llm(model=model, temperature=temperature)

    elif provider == "ollama":
        from models.ollama import get_ollama_llm
        return get_ollama_llm(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Provider không hỗ trợ: '{provider}'. Chọn: gemini | ollama")