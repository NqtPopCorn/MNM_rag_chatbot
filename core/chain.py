from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain(
    llm: BaseChatModel,
    retriever,
    prompt: ChatPromptTemplate,
) -> Runnable:
    """
    Lắp ráp RAG chain theo LCEL.
    Input: dict {"question": str, "chat_history": str}
    Output: chuỗi trả lời (str), hỗ trợ .stream()
    """
    return (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", "")
        }
        | prompt
        | llm
        | StrOutputParser()
    )