from typing import Annotated

from fastapi import APIRouter, Depends

from agrivibe.dependencies import get_chat_service
from agrivibe.models.chat import ChatRequest, ChatResponse
from agrivibe.service.chatting import ChatService

router = APIRouter(tags=["agents"])


@router.post("/chat")
async def chat(request: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]) -> ChatResponse:
    """Chat with AgriVibe. The orchestrator decides which specialist agents to call."""
    return await service.chat(request.message, request.user_id, request.thread_id)
