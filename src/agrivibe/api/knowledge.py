from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from agrivibe.core.vectorstore import Category
from agrivibe.dependencies import get_knowledge_service
from agrivibe.schemas import IndexResponse, SearchRequest, SearchResponse
from agrivibe.services.knowledge import KnowledgeService

router = APIRouter(tags=["knowledge"])

Service = Annotated[KnowledgeService, Depends(get_knowledge_service)]


@router.post("/documents", status_code=201)
def index_document(file: UploadFile, category: Annotated[Category, Form()], service: Service) -> IndexResponse:
    """Upload a PDF. It is split into chunks, embedded, and stored in Pinecone."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files can be indexed.")
    try:
        return service.index_pdf(file.filename, file.file.read(), category)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/search")
def search(request: SearchRequest, service: Service) -> SearchResponse:
    """Plain vector search, no model involved. Useful for checking what retrieval returns."""
    return service.search(request.query, request.k, request.category)
