import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agrivibe.api import ask, knowledge
from agrivibe.core.mcp import load_fieldsat
from agrivibe.dependencies import get_settings

# Print our own INFO lines, such as how long each rerank took. Libraries stay at WARNING.
logging.basicConfig(format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("agrivibe").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the app starts, before the first request.

    Asking an MCP server for its tools is async, so it happens here rather than in an
    lru_cache function in dependencies.py. Like the API keys, FieldSat is required:
    the app refuses to start if the server cannot be reached.
    """
    settings = get_settings()
    fieldsat = await load_fieldsat(settings.fieldsat_url)
    print("FieldSat tools:", ", ".join(tool.name for tool in fieldsat.tools))
    yield


app = FastAPI(
    title="FieldOracle",
    description="The AgriVibe knowledge service: index PDFs, then ask a RAG agent for cited answers.",
    version="0.1.0",
    lifespan=lifespan,
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
