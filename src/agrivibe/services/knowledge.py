import hashlib
import re
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from agrivibe.db.vectorstore import Category, search_chunks
from agrivibe.models.knowledge import IndexResponse, SearchHit, SearchResponse


def make_doc_id(filename: str, data: bytes) -> str:
    """Builds a stable id for a document from its name and contents.

    The same file always gets the same id, so re-uploading a PDF overwrites its
    chunks instead of duplicating them.

    Args:
        filename: The uploaded file's name.
        data: The file's bytes.

    Returns:
        The slugified file stem plus the first six hex digits of its SHA-256,
        for example `paddy-cultivation-guide-a29d87`.
    """
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    digest = hashlib.sha256(data).hexdigest()[:6]
    return f"{stem}-{digest}"


def read_pdf(data: bytes, metadata: dict) -> list[Document]:
    """Extracts the text of each page of a PDF.

    Args:
        data: The PDF's bytes.
        metadata: Fields copied onto every page, and from there onto every chunk.

    Returns:
        One document per page that has text, with its 1-based page number in
        the metadata. Pages without text (scans, blank pages) are skipped.
    """
    reader = PdfReader(BytesIO(data))
    return [
        Document(page_content=text, metadata={**metadata, "page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]


class KnowledgeService:
    """Indexes PDFs into the vector store and searches them."""

    def __init__(self, vector_store: VectorStore, chunk_size: int, chunk_overlap: int):
        """Initializes the service.

        Args:
            vector_store: Where chunks are stored and searched.
            chunk_size: The maximum characters per chunk.
            chunk_overlap: The characters shared by neighbouring chunks.
        """
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def index_pdf(self, filename: str, data: bytes, category: Category) -> IndexResponse:
        """Splits a PDF into chunks, tags each one and upserts them.

        Args:
            filename: The uploaded file's name, stored as each chunk's source.
            data: The PDF's bytes.
            category: The document class every chunk is tagged with.

        Returns:
            The document id and how many pages and chunks were indexed.

        Raises:
            ValueError: If no page of the PDF has extractable text.
        """
        doc_id = make_doc_id(filename, data)
        pages = read_pdf(data, {"doc_id": doc_id, "source": filename, "category": category})
        if not pages:
            raise ValueError("No text could be extracted. Is this a scanned PDF?")

        chunks = self.splitter.split_documents(pages)
        for number, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = number
            chunk.id = f"{doc_id}-{number}"

        self.vector_store.add_documents(chunks)
        return IndexResponse(doc_id=doc_id, filename=filename, category=category, pages=len(pages), chunks=len(chunks))

    def search(self, query: str, k: int, category: Category | None = None) -> SearchResponse:
        """Runs a plain vector search, with no model involved.

        Args:
            query: The text to search for.
            k: The number of hits to return.
            category: If set, only chunks tagged with this category are searched.

        Returns:
            The hits with their metadata and similarity scores, best match first.
        """
        hits = [
            SearchHit(
                id=doc.id,
                source=doc.metadata["source"],
                page=doc.metadata["page"],
                category=doc.metadata["category"],
                text=doc.page_content,
                score=score,
            )
            for doc, score in search_chunks(self.vector_store, query, k, category)
        ]
        return SearchResponse(hits=hits)
