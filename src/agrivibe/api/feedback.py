from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from agrivibe.dependencies import get_chat_service
from agrivibe.models.chat import FeedbackRequest
from agrivibe.service.chatting import ChatService

router = APIRouter(tags=["agents"])


@router.post("/feedback", status_code=204)
def feedback(request: FeedbackRequest, service: Annotated[ChatService, Depends(get_chat_service)]) -> None:
    """Record a farmer's 👍/👎 on one reply, as the `user_feedback` score on that reply's Langfuse trace."""
    if service.observability is None:
        raise HTTPException(503, "Feedback needs Langfuse, and the Langfuse keys are not set.")
    service.observability.feedback(request.trace_id, request.positive, request.comment)
