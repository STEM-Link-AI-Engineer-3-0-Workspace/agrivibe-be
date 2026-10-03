from typing import Annotated

from fastapi import APIRouter, Depends

from agrivibe.dependencies import get_answer_service
from agrivibe.schemas import AskRequest, AskResponse
from agrivibe.services.answering import AnswerService

router = APIRouter(tags=["agents"])


@router.post("/ask")
def ask(request: AskRequest, service: Annotated[AnswerService, Depends(get_answer_service)]) -> AskResponse:
    """Ask the RAG agent. Returns the answer as sentence and citation pairs."""
    return service.ask(request.question)
