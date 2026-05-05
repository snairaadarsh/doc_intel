"""
rag.py
──────
Document Intelligence System — RAG backend
Supports: PDF, TXT, DOCX/DOC, CSV, XLSX/XLS, PPTX, MD, HTML
Vector store: FAISS (local, persistent)
LLM: Groq llama-3.3-70b-versatile (free tier)
Embeddings: HuggingFace gte-base-en-v1.5 (local, free)
"""

from __future__ import annotations

import os
import csv
import tempfile
from pathlib import Path
from typing import Optional, Callable

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 100
EMBEDDING_MODEL   = "Alibaba-NLP/gte-base-en-v1.5"
FAISS_INDEX_DIR   = Path(__file__).parent / "resource" / "faiss_index"

LLM_MODEL         = "llama-3.3-70b-versatile"
MAX_ANSWER_TOKENS = 600
LLM_CONTEXT_LIMIT = 4096
SYSTEM_TOKENS     = 250
CONTEXT_BUDGET    = LLM_CONTEXT_LIMIT - MAX_ANSWER_TOKENS - SYSTEM_TOKENS
CHARS_PER_TOKEN   = 4
RETRIEVAL_K       = 12

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt",
    ".docx", ".doc",
    ".csv",
    ".xlsx", ".xls",
    ".pptx",
    ".md", ".markdown",
    ".html", ".htm",
}

# ── Globals ───────────────────────────────────────────────────────────────────
_llm:          Optional[ChatGroq]              = None
_embeddings:   Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS]                 = None
_indexed_files: list[str]                      = []


# ── Internal helpers ──────────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def select_chunks_within_budget(
    docs_and_scores: list[tuple[Document, float]], budget: int
) -> list[tuple[Document, float]]:
    sorted_pairs = sorted(docs_and_scores, key=lambda x: x[1])
    selected, used = [], 0
    for doc, score in sorted_pairs:
        tokens = estimate_tokens(doc.page_content)
        if used + tokens > budget:
            continue
        selected.append((doc, score))
        used += tokens
    return selected or [min(docs_and_scores, key=lambda x: x[1])]


def build_context(selected: list[tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(selected, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        page_s = f" | page {page + 1}" if page != "" else ""
        parts.append(
            f"[Chunk {i} | score={score:.4f} | {source}{page_s}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )
    return _embeddings


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.7,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    return _llm


# ── Format-specific loaders ───────────────────────────────────────────────────
def _load_csv(path: str) -> list[Document]:
    """
    Load a CSV file. Each row becomes its own Document so
    column context is preserved.
    """
    docs = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": Path(path).name, "row": i},
                ))
    return docs


def _load_excel(path: str) -> list[Document]:
    """
    Load XLSX / XLS. Each sheet's rows become Documents.
    Requires openpyxl (xlsx) / xlrd (xls).
    """
    import openpyxl

    suffix = Path(path).suffix.lower()
    docs   = []

    if suffix == ".xlsx":
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet_name in wb.sheetnames:
            ws      = wb[sheet_name]
            headers = None
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                row = [str(c) if c is not None else "" for c in row]
                if i == 0:
                    headers = row
                    continue
                if not any(row):
                    continue
                if headers:
                    text = "\n".join(
                        f"{h}: {v}" for h, v in zip(headers, row) if v
                    )
                else:
                    text = "  ".join(row)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": Path(path).name,
                            "sheet": sheet_name,
                            "row": i,
                        },
                    ))
        wb.close()

    elif suffix == ".xls":
        import xlrd
        wb = xlrd.open_workbook(path)
        for sheet in wb.sheets():
            headers = None
            for i in range(sheet.nrows):
                row = [str(sheet.cell_value(i, j)) for j in range(sheet.ncols)]
                if i == 0:
                    headers = row
                    continue
                if not any(row):
                    continue
                if headers:
                    text = "\n".join(
                        f"{h}: {v}" for h, v in zip(headers, row) if v
                    )
                else:
                    text = "  ".join(row)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": Path(path).name,
                            "sheet": sheet.name,
                            "row": i,
                        },
                    ))

    return docs


def _load_file(path: str) -> list[Document]:
    """
    Route a file to the correct loader based on its extension.
    Returns a list of LangChain Documents.
    """
    suffix = Path(path).suffix.lower()
    fname  = Path(path).name

    if suffix == ".pdf":
        loader = PyPDFLoader(path)
        docs   = loader.load()

    elif suffix == ".txt":
        loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
        docs   = loader.load()

    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(path)
        docs   = loader.load()

    elif suffix == ".csv":
        docs = _load_csv(path)

    elif suffix in (".xlsx", ".xls"):
        docs = _load_excel(path)

    elif suffix == ".pptx":
        loader = UnstructuredPowerPointLoader(path)
        docs   = loader.load()

    elif suffix in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(path)
        docs   = loader.load()

    elif suffix in (".html", ".htm"):
        loader = UnstructuredHTMLLoader(path)
        docs   = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Normalise source metadata to filename only
    for doc in docs:
        doc.metadata["source"] = fname

    return docs


# ── Public API ────────────────────────────────────────────────────────────────
def process_documents(
    file_paths: list[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Ingest a batch of documents into a FAISS vector store.

    Args:
        file_paths:        Local file paths (up to 100 recommended).
        progress_callback: Optional callable(current, total, filename).

    Returns:
        {total_files, total_chunks, skipped, indexed_files}
    """
    global _vector_store, _indexed_files

    _get_llm()          # warm up
    ef = _get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_docs:     list[Document] = []
    loaded_files: list[str]      = []
    skipped:      list[str]      = []
    total                        = len(file_paths)

    for idx, fp in enumerate(file_paths):
        fname = Path(fp).name
        if progress_callback:
            progress_callback(idx, total, fname)
        try:
            docs = _load_file(fp)
            all_docs.extend(docs)
            loaded_files.append(fname)
            print(f"  [ok] {fname} → {len(docs)} doc(s)")
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
            skipped.append(fname)

    if not all_docs:
        raise ValueError(
            "No documents could be loaded. "
            "Check that the files are valid and non-empty."
        )

    chunks = splitter.split_documents(all_docs)
    print(f"[index] {len(all_docs)} docs → {len(chunks)} chunks")

    # Build FAISS index and persist
    _vector_store  = FAISS.from_documents(chunks, ef)
    _indexed_files = loaded_files

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _vector_store.save_local(str(FAISS_INDEX_DIR))

    if progress_callback:
        progress_callback(total, total, "Done")

    return {
        "total_files":   len(loaded_files),
        "total_chunks":  len(chunks),
        "skipped":       skipped,
        "indexed_files": loaded_files,
    }


def load_existing_index() -> bool:
    """Load a previously saved FAISS index from disk."""
    global _vector_store, _indexed_files
    index_path = FAISS_INDEX_DIR / "index.faiss"
    if not index_path.exists():
        return False
    try:
        ef            = _get_embeddings()
        _vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            ef,
            allow_dangerous_deserialization=True,
        )
        return True
    except Exception as e:
        print(f"[warn] Could not load existing index: {e}")
        return False


def generate_answer(query: str) -> tuple[str, str, list[dict]]:
    """
    Query the indexed documents.

    Returns:
        (answer_str, sources_str, chunk_metadata_list)
    """
    if _vector_store is None:
        raise RuntimeError("No index loaded. Process documents first.")

    docs_and_scores = _vector_store.similarity_search_with_score(query, k=RETRIEVAL_K)
    if not docs_and_scores:
        return "No relevant information found in the indexed documents.", "", []

    selected     = select_chunks_within_budget(docs_and_scores, CONTEXT_BUDGET)
    total_tokens = sum(estimate_tokens(d.page_content) for d, _ in selected)
    print(
        f"[RAG] retrieved={len(docs_and_scores)}  "
        f"selected={len(selected)}  tokens≈{total_tokens}/{CONTEXT_BUDGET}"
    )

    context        = build_context(selected)
    unique_sources = sorted({doc.metadata.get("source", "?") for doc, _ in selected})
    sources_str    = ", ".join(unique_sources)

    chunk_meta = [
        {
            "chunk":  i + 1,
            "source": doc.metadata.get("source", "?"),
            "page":   doc.metadata.get("page", ""),
            "score":  round(float(score), 4),
            "tokens": estimate_tokens(doc.page_content),
        }
        for i, (doc, score) in enumerate(selected)
    ]

    prompt = (
        "You are a precise document analysis assistant. "
        "Answer the question below using ONLY the context provided. "
        "If the answer is not in the context, clearly say so. "
        "Be concise and structured in your response.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    llm      = _get_llm()
    response = llm.invoke(prompt)
    answer   = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), sources_str, chunk_meta


def get_index_stats() -> dict:
    if _vector_store is None:
        return {"status": "empty", "total_vectors": 0, "indexed_files": []}
    try:
        n = _vector_store.index.ntotal
    except Exception:
        n = "unknown"
    return {
        "status":        "ready",
        "total_vectors": n,
        "indexed_files": _indexed_files,
    }


def is_ready() -> bool:
    return _vector_store is not None
