from typing import Literal, List

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore: FAISS,
    k: int = 5,
    score_threshold: float = 0.3,
) -> VectorStoreRetriever:
    """
    Standard vector similarity retriever.

    Parameters
    ----------
    k               : max chunks to return
    score_threshold : minimum cosine-similarity score (0.0 → 1.0)
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def get_hybrid_retriever(
    vectorstore: FAISS,
    k: int = 5,
    score_threshold: float = 0.3,
):
    """
    Hybrid retriever: Reciprocal Rank Fusion of vector search + BM25.

    How it works
    ------------
    1. Vector retriever finds semantically similar chunks (dense retrieval).
    2. BM25 retriever finds keyword-matching chunks (sparse retrieval).
    3. Both ranked lists are merged via RRF: score = Σ 1/(60 + rank).
       Chunks appearing in both lists naturally score higher.

    All documents are pulled from the FAISS docstore so BM25 can index them
    — no separate document list needs to be maintained.

    Requires:  pip install rank_bm25
    """
    from core.hybrid_retriever import HybridRetriever  # local module

    all_docs = list(vectorstore.docstore._dict.values())
    vector_retriever = get_retriever(vectorstore, k=k, score_threshold=score_threshold)
    return HybridRetriever(vector_retriever=vector_retriever, documents=all_docs, k=k)


def build_retriever(
    vectorstore: FAISS,
    retriever_type: Literal["vector", "hybrid"] = "vector",
    k: int = 5,
    score_threshold: float = 0.3,
):
    """
    Factory that returns the right retriever based on *retriever_type*.

    Parameters
    ----------
    retriever_type : "vector" | "hybrid"
    """
    if retriever_type == "hybrid":
        return get_hybrid_retriever(vectorstore, k=k, score_threshold=score_threshold)
    return get_retriever(vectorstore, k=k, score_threshold=score_threshold)