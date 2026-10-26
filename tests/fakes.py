"""Stand-ins for OpenAI, Pinecone and FieldSat, so the tests run offline and for free."""

import warnings

from fastmcp import FastMCP
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import InMemoryVectorStore

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain.mcp import MCPAdapter


class FakeVectorStore(InMemoryVectorStore):
    """In-memory store that accepts Pinecone-style {"key": value} filters."""

    def __init__(self):
        super().__init__(DeterministicFakeEmbedding(size=64))

    def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        if isinstance(filter, dict):
            wanted = filter

            def matches(doc):
                return all(doc.metadata.get(key) == value for key, value in wanted.items())

            filter = matches

        return super().similarity_search_with_score(query, k=k, filter=filter, **kwargs)


class ScriptedChatModel(BaseChatModel):
    """Replies with a fixed list of messages, one per call. Tool calls included."""

    replies: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.replies[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FakeReranker:
    """Stands in for the cross-encoder. A chunk scores one point per query word it contains."""

    kind = "fake"

    def __init__(self, candidates=20):
        self.candidates = candidates
        self.calls = 0

    def score(self, query, docs):
        self.calls += 1
        words = set(query.lower().split())
        return [float(sum(word in doc.page_content.lower() for word in words)) for doc in docs]


def fake_fieldsat() -> FastMCP:
    """A stand-in for the FieldSat MCP server: the same three tools, fixed numbers.

    MCPAdapter connects to it in memory, so tests need no server process. The numbers
    are FieldSat's demo values for the Minneriya field.
    """
    mcp = FastMCP("FieldSat (test)")

    @mcp.tool
    def list_fields() -> dict:
        """List every field this server can analyse."""
        return {"fields": [{"id": "SL-PADDY-MINNERIYA", "name": "Minneriya Paddy Fields", "crop": "paddy"}]}

    @mcp.tool
    def get_satellite_analysis(field_id: str, comparison_days: int) -> dict:
        """Sentinel-2 vegetation change (NDVI) and data quality for one registered field."""
        return {
            "field_id": field_id,
            "current_mean_ndvi": 0.56,
            "previous_mean_ndvi": 0.66,
            "ndvi_change_percentage": -15.2,
            "trend": "significant_decline",
            "ndvi_standard_deviation": 0.17,
            "field_uniformity": "uneven",
            "valid_observations": 4,
            "average_cloud_coverage_percentage": 11.0,
            "data_quality": "good",
        }

    @mcp.tool
    def get_weather_context(field_id: str) -> dict:
        """Recent rainfall and the 48-hour rain forecast for one registered field."""
        return {
            "rainfall_last_7_days_mm": 112.0,
            "rainy_days_last_7_days": 5,
            "rainfall_last_48_hours_mm": 38.0,
            "rainfall_forecast_next_48_hours_mm": 46.0,
            "average_humidity_last_7_days": 88.0,
            "heavy_rain_recent": True,
            "heavy_rain_expected": True,
        }

    return mcp


async def load_fake_fieldsat_tools() -> list:
    """The fake server's tools, exactly as MCPAdapter hands them to an agent."""
    async with MCPAdapter(fake_fieldsat()) as adapter:
        return await adapter.list_tools()
