from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from agrivibe.dependencies import current_farmer, get_chat_service
from agrivibe.models.chat import ChatRequest, ChatResponse
from agrivibe.models.farmer import Farmer
from agrivibe.service.chatting import ChatService
from agrivibe.service.threads import ThreadNotFound

router = APIRouter(tags=["agents"])


@router.post("/chat", responses={404: {"description": "thread_id is not one of your conversations."}})
async def chat(
    request: ChatRequest,
    farmer: Annotated[Farmer, Depends(current_farmer)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Chat with AgriVibe. The orchestrator decides which specialist agents to call."""
    try:
        return await service.chat(request.message, farmer.id, request.thread_id)
    except ThreadNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None
