import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n",
    "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )


# =========================
# LOAD VECTORSTORE
# =========================
def load_vectorstore(db_path: str, embeddings: Embeddings) -> FAISS | None:
    if not os.path.exists(db_path):
        return None

    return FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


# =========================
# HELPER: ADD PAGE INTO CONTENT
# =========================
def _inject_page_info(documents: List[Document]) -> List[Document]:
    for doc in documents:
        page = doc.metadata.get("page", "unknown")
        source = os.path.basename(doc.metadata.get("source", ""))

        doc.page_content = f"[Source: {source} | Page: {page}]\n{doc.page_content}"

    return documents


# =========================
# ADD DOCUMENTS
# =========================
def add_documents_to_db(
    file_paths: List[str],
    db_path: str,
    embeddings: Embeddings,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Tuple[bool, str]:
    try:
        documents: List[Document] = []

        # 🔥 Load từng file với PyPDFLoader
        for fp in file_paths:
            loader = PyPDFLoader(fp)
            docs = loader.load()

            # inject page info
            docs = _inject_page_info(docs)

            documents.extend(docs)

        if not documents:
            return False, "Không đọc được nội dung file."

        # 🔥 Split
        splits = _make_splitter(chunk_size, chunk_overlap).split_documents(documents)

        # 🔥 Load existing DB
        existing = load_vectorstore(db_path, embeddings)

        if existing:
            existing.add_documents(splits)
            existing.save_local(db_path)
        else:
            vs = FAISS.from_documents(documents=splits, embedding=embeddings)
            vs.save_local(db_path)

        return True, f"✅ Đã thêm {len(splits)} chunks vào Database."

    except Exception as e:
        return False, f"❌ Lỗi: {e}"


# =========================
# REBUILD DATABASE
# =========================
def rebuild_db(
    papers_dir: str,
    db_path: str,
    embeddings: Embeddings,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Tuple[bool, str]:
    try:
        all_docs: List[Document] = []

        # 🔥 Load toàn bộ PDF trong thư mục
        for root, _, files in os.walk(papers_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    fp = os.path.join(root, file)

                    loader = PyPDFLoader(fp)
                    docs = loader.load()

                    docs = _inject_page_info(docs)
                    all_docs.extend(docs)

        if not all_docs:
            return False, "Thư mục không có file PDF nào."

        splits = _make_splitter(chunk_size, chunk_overlap).split_documents(all_docs)

        vs = FAISS.from_documents(documents=splits, embedding=embeddings)
        vs.save_local(db_path)

        return True, f"✅ Build lại hoàn tất với {len(splits)} chunks."

    except Exception as e:
        return False, f"❌ Lỗi: {e}"


# =========================
# GET ALL DOCUMENTS
# =========================
def get_all_documents(vectorstore: FAISS) -> List[Document]:
    return list(vectorstore.docstore._dict.values())


# =========================
# DEBUG / TEST RETRIEVE
# =========================
def debug_retrieve(vectorstore: FAISS, query: str, k: int = 5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    for i, d in enumerate(docs):
        print(f"\n=== Result {i+1} ===")
        print("Source:", d.metadata.get("source"))
        print("Page:", d.metadata.get("page"))
        print("Content preview:", d.page_content[:300])