from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore: FAISS,
    k: int = 5,
    score_threshold: float = 0.3,
) -> VectorStoreRetriever:
    """
    Tạo retriever với similarity_score_threshold.
    - k: số lượng chunk tối đa trả về
    - score_threshold: ngưỡng tương đồng tối thiểu (0.0 → 1.0)
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )