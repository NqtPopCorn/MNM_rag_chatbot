from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate


def build_rag_chain(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    prompt: ChatPromptTemplate,
) -> Runnable:
    """
    Lắp ráp RAG chain theo LCEL.
    Input: câu hỏi (str)
    Output: chuỗi trả lời (str), hỗ trợ .stream()
    """
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )