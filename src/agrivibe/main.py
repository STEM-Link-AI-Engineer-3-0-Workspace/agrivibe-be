import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agrivibe.api import ask, knowledge
from agrivibe.dependencies import get_settings

# Print our own INFO lines, such as how long each rerank took. Libraries stay at WARNING.
logging.basicConfig(format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("agrivibe").setLevel(logging.INFO)

app = FastAPI(
    title="FieldOracle",
    description="The AgriVibe knowledge service: index PDFs, then ask a RAG agent for cited answers.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge.router)
app.include_router(ask.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Reports that the service is up."""
    return {"status": "ok"}
