from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GEMINI_LLM_MODELS = [
    "gemini-2.5-flash-lite",
]

GEMINI_EMBEDDING_MODELS = [
    "gemini-embedding-001",
]


def get_gemini_llm(model: str = "gemini-2.5-flash-lite", temperature: float = 0.1):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def get_gemini_embeddings(model: str = "gemini-embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model)