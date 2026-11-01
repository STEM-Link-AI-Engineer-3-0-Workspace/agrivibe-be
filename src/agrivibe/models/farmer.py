"""Body for GET /me."""

from pydantic import BaseModel, Field


class Farmer(BaseModel):
    """The farmer the app is acting for."""

    id: str = Field(description="An opaque id, the kind an auth service issues.", examples=["frm_7f3a9c2e"])
    name: str
