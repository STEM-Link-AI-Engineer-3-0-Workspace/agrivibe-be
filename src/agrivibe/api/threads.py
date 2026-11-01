from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from agrivibe.dependencies import current_farmer, get_thread_service
from agrivibe.models.farmer import Farmer
from agrivibe.models.threads import Thread, ThreadSummary
from agrivibe.service.threads import ThreadNotFound, ThreadService

router = APIRouter(prefix="/threads", tags=["threads"])

NOT_FOUND = {404: {"description": "Not one of your conversations."}}


@router.get("")
async def list_threads(
    farmer: Annotated[Farmer, Depends(current_farmer)],
    service: Annotated[ThreadService, Depends(get_thread_service)],
) -> list[ThreadSummary]:
    """Your conversations, the most recently used first."""
    return await service.list(farmer.id)


@router.get("/{thread_id}", responses=NOT_FOUND)
async def get_thread(
    thread_id: str,
    farmer: Annotated[Farmer, Depends(current_farmer)],
    service: Annotated[ThreadService, Depends(get_thread_service)],
) -> Thread:
    """One of your conversations: every message, its answer and the evidence behind it."""
    try:
        return await service.get(farmer.id, thread_id)
    except ThreadNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None


@router.delete("/{thread_id}", status_code=204, responses=NOT_FOUND)
async def delete_thread(
    thread_id: str,
    farmer: Annotated[Farmer, Depends(current_farmer)],
    service: Annotated[ThreadService, Depends(get_thread_service)],
) -> Response:
    """Deletes one of your conversations, with its history and evidence."""
    try:
        await service.delete(farmer.id, thread_id)
    except ThreadNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None
    return Response(status_code=204)
