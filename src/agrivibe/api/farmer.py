from typing import Annotated

from fastapi import APIRouter, Depends

from agrivibe.dependencies import current_farmer
from agrivibe.models.farmer import Farmer

router = APIRouter(tags=["farmer"])


@router.get("/me")
def me(farmer: Annotated[Farmer, Depends(current_farmer)]) -> Farmer:
    """Who the server thinks is asking."""
    return farmer
