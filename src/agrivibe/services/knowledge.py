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


def read_pdf(data: bytes) -> list[Document]:
    """Extracts the text of each page of a PDF.

    Args:
        data: The PDF's bytes.

    Returns:
        One document per page that has text, with its 1-based page number in
        the metadata. Pages without text (scans, blank pages) are skipped.
    """
    reader = PdfReader(BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Document(page_content=text, metadata={"page": number}))
    return pages


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
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
        pages = read_pdf(data)
        if not pages:
            raise ValueError("No text could be extracted. Is this a scanned PDF?")

        doc_id = make_doc_id(filename, data)
        chunks = self.splitter.split_documents(pages)
        ids = []
        for number, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-{number}"
            chunk.metadata.update(chunk_id=chunk_id, doc_id=doc_id, source=filename, category=category)
            ids.append(chunk_id)

        self.vector_store.add_documents(chunks, ids=ids)
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
                id=doc.metadata["chunk_id"],
                source=doc.metadata["source"],
                page=doc.metadata["page"],
                category=doc.metadata["category"],
                text=doc.page_content,
                score=score,
            )
            for doc, score in search_chunks(self.vector_store, query, k, category)
        ]
        return SearchResponse(hits=hits)
