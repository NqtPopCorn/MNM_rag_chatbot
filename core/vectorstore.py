import os
from typing import List, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
    """Load DB from disk. Returns None if it doesn't exist yet."""
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
    loader_strategy: Literal["standard", "enhanced"] = "standard",
) -> tuple[bool, str]:
    """
    Load new files → chunk → append to existing DB (or create one).

    Parameters
    ----------
    loader_strategy : "standard" | "enhanced"
        • "standard"  — PyPDFLoader (fast, plain text)
        • "enhanced"  — pymupdf4llm Markdown extraction (tables + images)

    Returns (success, message).
    """
    try:
        documents: List[Document] = []

        if loader_strategy == "enhanced":
            from core.multimodal_loader import load_pdf
            for fp in file_paths:
                documents.extend(load_pdf(fp, strategy="enhanced"))
        else:
            for fp in file_paths:
                loader = PyPDFLoader(fp)
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

        strategy_label = "Enhanced (pymupdf4llm)" if loader_strategy == "enhanced" else "Standard"
        return True, f"✅ Đã thêm {len(splits)} chunks [{strategy_label}] vào Database."
    except Exception as e:
        return False, f"❌ Lỗi: {e}"


def rebuild_db(
    papers_dir: str,
    db_path: str,
    embeddings: Embeddings,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    loader_strategy: Literal["standard", "enhanced"] = "standard",
) -> tuple[bool, str]:
    """
    Re-read the entire papers directory and rebuild the DB from scratch.

    Supports the same *loader_strategy* options as add_documents_to_db.
    """
    try:
        pdf_files = [
            os.path.join(papers_dir, f)
            for f in os.listdir(papers_dir)
            if f.lower().endswith(".pdf")
        ] if os.path.exists(papers_dir) else []

        if not pdf_files:
            return False, "Thư mục không có file PDF nào."

        if loader_strategy == "enhanced":
            from core.multimodal_loader import load_pdf
            docs: List[Document] = []
            for fp in pdf_files:
                docs.extend(load_pdf(fp, strategy="enhanced"))
        else:
            loader = DirectoryLoader(
                path=papers_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                use_multithreading=True,
            )
            docs = loader.load()

        if not docs:
            return False, "Không đọc được nội dung từ các file PDF."

        splits = _make_splitter(chunk_size, chunk_overlap).split_documents(docs)
        vs = FAISS.from_documents(documents=splits, embedding=embeddings)
        vs.save_local(db_path)

        strategy_label = "Enhanced (pymupdf4llm)" if loader_strategy == "enhanced" else "Standard"
        return True, f"✅ Build lại hoàn tất: {len(splits)} chunks [{strategy_label}]."
    except Exception as e:
        return False, f"❌ Lỗi: {e}"