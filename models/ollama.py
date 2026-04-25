from langchain_ollama import ChatOllama, OllamaEmbeddings

OLLAMA_LLM_MODELS = [
    "qwen2.5:3b",
    "gemma4:31b-cloud",
    "kimi-k2.6:cloud",
    "qwen3.5:397b-cloud"
]

OLLAMA_EMBEDDING_MODELS = [
    "nomic-embed-text",
]


def get_ollama_llm(model: str = "llama3.2", base_url: str = "http://localhost:11434", temperature: float = 0.1):
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def get_ollama_embeddings(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    return OllamaEmbeddings(model=model, base_url=base_url)