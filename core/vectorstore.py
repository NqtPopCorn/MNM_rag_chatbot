import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
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


def load_vectorstore(db_path: str, embeddings: Embeddings) -> FAISS | None:
    """Load DB từ disk. Trả về None nếu chưa tồn tại."""
    if not os.path.exists(db_path):
        return None
    return FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def add_documents_to_db(
    file_paths: List[str],
    db_path: str,
    embeddings: Embeddings,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> tuple[bool, str]:
    """
    Load các file mới → chunk → append vào DB hiện tại (hoặc tạo mới nếu chưa có).
    Trả về (success: bool, message: str).
    """
    try:
        documents: List[Document] = []
        for fp in file_paths:
            loader = UnstructuredFileLoader(fp)
            documents.extend(loader.load())

        if not documents:
            return False, "Không đọc được nội dung file."

        splits = _make_splitter(chunk_size, chunk_overlap).split_documents(documents)

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


def rebuild_db(
    papers_dir: str,
    db_path: str,
    embeddings: Embeddings,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> tuple[bool, str]:
    """Đọc lại TOÀN BỘ thư mục papers → tạo lại DB từ đầu."""
    try:
        loader = DirectoryLoader(
            path=papers_dir,
            glob="**/*.pdf",
            loader_cls=UnstructuredFileLoader,
            use_multithreading=True,
        )
        docs = loader.load()
        if not docs:
            return False, "Thư mục không có file PDF nào."

        splits = _make_splitter(chunk_size, chunk_overlap).split_documents(docs)
        vs = FAISS.from_documents(documents=splits, embedding=embeddings)
        vs.save_local(db_path)
        return True, f"✅ Build lại hoàn tất với {len(splits)} chunks."
    except Exception as e:
        return False, f"❌ Lỗi: {e}"