import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agrivibe.api import chat, farmer, knowledge, threads
from agrivibe.core.database import open_database
from agrivibe.core.mcp import load_fieldsat
from agrivibe.dependencies import build_chat_service, get_settings
from agrivibe.models.farmer import Farmer
from agrivibe.service.threads import ThreadService

# Print our own INFO lines, such as how long each rerank took. Libraries stay at WARNING.
logging.basicConfig(format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("agrivibe").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the app starts, before the first request.

    Asking an MCP server for its tools and opening Postgres are async, so they happen here
    rather than in lru_cache functions in dependencies.py. Like the API keys, FieldSat and
    Postgres are required: the app refuses to start if either cannot be reached.
    """
    settings = get_settings()
    fieldsat = await load_fieldsat(settings.fieldsat_url)
    print("FieldSat tools:", ", ".join(tool.name for tool in fieldsat.tools))
    async with open_database(settings.database_url.get_secret_value()) as db:
        print("Postgres: connected; threads and farmer memory persist")
        app.state.thread_service = ThreadService(db.pool, db.checkpointer)
        # The seed: the one farmer this deployment serves (see current_farmer) must exist before they chat.
        await app.state.thread_service.add_farmer(Farmer(id=settings.farmer_id, name=settings.farmer_name))
        app.state.chat_service = build_chat_service(
            settings, fieldsat, db.checkpointer, db.store, app.state.thread_service
        )
        yield


app = FastAPI(
    title="AgriVibe",
    description="The AgriVibe backend: a multi-agent farm advisor with cited answers, field evidence from FieldSat over MCP, and conversations kept in Postgres.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(threads.router)
app.include_router(farmer.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Reports that the service is up."""
    return {"status": "ok"}
