from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from collections import defaultdict


class HybridRetriever:
    """
        Using for Hyrid search between vector retriever and bm25 retriver with basic ranking 
    """

    def __init__(self, vector_retriever, documents: List[Document], k=5):
        self.vector_retriever = vector_retriever
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = k
        self.k = k

    def invoke(self, query: str):
        vec_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25.invoke(query)

        return self.merge(vec_docs, bm25_docs)

    def merge(self, vec_docs, bm25_docs):
        """vec_docs, bm25_docs are both sorted"""

        scores = defaultdict(float)

        # vector ranking
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content
            scores[key] += 1 / (60 + rank)

        # bm25 ranking
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] += 1 / (60 + rank)

        # đeduplicate
        doc_map = {}
        for doc in vec_docs + bm25_docs:
            doc_map[doc.page_content] = doc

        # sort theo score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[key] for key, _ in sorted_docs[:self.k]]

    def __call__(self, query: str):
        return self.invoke(query)