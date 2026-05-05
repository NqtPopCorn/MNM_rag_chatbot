"""
Multimodal PDF Loader
=====================
Handles PDFs that contain tables and/or images using two strategies:

  • standard   — UnstructuredFileLoader (current behaviour, fast)
  • enhanced   — pymupdf4llm: converts each page to Markdown so tables are
                 preserved as Markdown tables and text layout is kept intact.
                 Falls back to standard automatically if pymupdf4llm is absent.

Public API
----------
  load_pdf(file_path, strategy="standard") -> List[Document]
  LOADER_DISPLAY_NAMES   # dict for sidebar radio labels
"""

import os
from typing import List

from langchain_core.documents import Document

# Human-readable labels used in the Streamlit sidebar
LOADER_DISPLAY_NAMES = {
    "Standard": "standard",
    "Enhanced — Tables & Images": "enhanced",
}


# ── Strategy: standard ────────────────────────────────────────────────────────

def _load_standard(file_path: str) -> List[Document]:
    """Original UnstructuredFileLoader — fast, works for plain-text PDFs."""
    from langchain_community.document_loaders import UnstructuredFileLoader
    return UnstructuredFileLoader(file_path).load()


# ── Strategy: enhanced (pymupdf4llm) ─────────────────────────────────────────

def _load_enhanced(file_path: str) -> List[Document]:
    """
    Use pymupdf4llm to render each PDF page as Markdown.

    Why Markdown output?
    • Tables  → preserved as Markdown tables  (```| col | col |```)
    • Headers → preserved with # notation
    • Columns → merged in reading order (not garbled left/right columns)

    Images in the PDF are skipped by default (describing them requires a
    vision model; enable by setting write_images=True and adding a captioning
    step after loading).

    Requires:  pip install pymupdf4llm
    """
    try:
        import pymupdf4llm  # type: ignore
    except ImportError:
        print("[multimodal_loader] pymupdf4llm not installed — falling back to standard loader.")
        return _load_standard(file_path)

    # page_chunks=True → one dict per page: {"text": "...", "metadata": {...}}
    pages = pymupdf4llm.to_markdown(
        doc=file_path,
        page_chunks=True,   # one Document per page
        write_images=False, # set True to also dump embedded images to disk
    )

    docs: List[Document] = []
    for page_data in pages:
        text = page_data.get("text", "").strip()
        if not text:
            continue

        # pymupdf4llm nests metadata inside "metadata" key
        inner_meta = page_data.get("metadata", {})
        docs.append(Document(
            page_content=text,
            metadata={
                "source": file_path,
                "page": inner_meta.get("page", page_data.get("page", "?")),
                "loader": "pymupdf4llm",
            },
        ))
    return docs


# ── Public helper ─────────────────────────────────────────────────────────────

def load_pdf(file_path: str, strategy: str = "standard") -> List[Document]:
    """
    Load a single PDF file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF.
    strategy  : "standard" | "enhanced"
        • "standard"  — UnstructuredFileLoader (default, no extra deps)
        • "enhanced"  — pymupdf4llm Markdown extraction (better for tables/images)
    """
    if strategy == "enhanced":
        return _load_enhanced(file_path)
    return _load_standard(file_path)