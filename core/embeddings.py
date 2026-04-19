from typing import Literal
from langchain_core.embeddings import Embeddings
from config.settings import settings


def embedding_factory(
    provider: Literal["gemini", "ollama"],
    model: str,
) -> Embeddings:
    """
    LƯU Ý: Embedding provider/model phải NHẤT QUÁN giữa lúc build DB và lúc query.
    Nếu đổi provider hoặc model → cần build lại toàn bộ DB.
    """
    if provider == "gemini":
        from models.gemini import get_gemini_embeddings
        return get_gemini_embeddings(model=model)

    elif provider == "ollama":
        from models.ollama import get_ollama_embeddings
        return get_ollama_embeddings(model=model, base_url=settings.ollama_base_url)

    else:
        raise ValueError(f"Embedding provider không hỗ trợ: '{provider}'. Chọn: gemini | ollama")