from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from agrivibe.db.vectorstore import Category
from agrivibe.dependencies import get_knowledge_service
from agrivibe.models.knowledge import IndexResponse, SearchRequest, SearchResponse
from agrivibe.service.knowledge import KnowledgeService

router = APIRouter(tags=["knowledge"])

Service = Annotated[KnowledgeService, Depends(get_knowledge_service)]


@router.post("/documents", status_code=201)
def index_document(file: UploadFile, category: Annotated[Category, Form()], service: Service) -> IndexResponse:
    """Upload a PDF. It is split into chunks, embedded, and stored in Pinecone."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files can be indexed.")
    try:
        return service.index_pdf(filename, file.file.read(), category)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/search")
def search(request: SearchRequest, service: Service) -> SearchResponse:
    """Vector search, then rerank. No chat model involved. Useful for checking what retrieval returns."""
    return service.search(request.query, request.k, request.category, request.rerank)
