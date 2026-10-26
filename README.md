# FieldOracle

The knowledge service behind AgriVibe. You upload PDFs (cultivation guides, government
circulars, farm records) and FieldOracle splits them, embeds them and stores them in
Pinecone. You ask a question and a RAG agent searches that store and answers with a
citation after every sentence, so every sentence traces back to passages in a PDF.

It is a FastAPI app built in Level 4 Lesson 2 of the STEMLink AI Engineer Bootcamp. The
code is kept small on purpose, but the layout is the one you would use at work, and it is
the backend the AgriVibe frontend will call later.

## Setup

You need an OpenAI API key and a Pinecone API key (the free Starter plan is enough).

### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
uv sync
cp .env.example .env
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # skip if you already have uv
uv sync
copy .env.example .env
```

Then open `.env` and paste in both keys.

The app also needs the FieldSat MCP server from Level 6 Lesson 3. In that repo, in a second
terminal, start it before you start this app:

```bash
uv run fieldsat --demo --transport http --port 8001
```

The code is a package, `agrivibe`, in `src/`. `uv sync` installs it into `.venv` in
editable mode, so `import agrivibe` works from anywhere and your edits apply without a
reinstall.

## Run

```bash
uv run fastapi dev
```

Open <http://127.0.0.1:8000/docs>. Every endpoint can be tried from that page.

## Build log

Each step is one branch and one pull request, merged in class.

### Step 01 · The app skeleton

`src/agrivibe/config.py` reads every setting from `.env` with pydantic-settings. The two API keys
have no default, so the app refuses to start without them. `src/agrivibe/main.py` creates the
FastAPI app, allows the AgriVibe frontend's origin (CORS), and serves `GET /health`.
`src/agrivibe/dependencies.py` is where every shared object gets built. It holds one function so
far.

### Step 02 · The vector store

`src/agrivibe/db/vectorstore.py` connects to Pinecone and creates the index on first run with
1536 dimensions and the cosine metric, which match `text-embedding-3-small`. It also defines
`Category`, the three document classes every chunk is tagged with, and `search_chunks`,
the one search function the rest of the app uses.

### Step 03 · Indexing and search

`POST /documents` takes a PDF and a category. `KnowledgeService.index_pdf` reads it page by
page with pypdf, splits it with `RecursiveCharacterTextSplitter`, tags each chunk with
its source, page, category and an id, then upserts it to Pinecone.

The id is the file name plus a hash of the file's bytes, plus the chunk number, for example
`paddy-cultivation-guide-a29d87-2`. Uploading the same PDF twice overwrites the same
ids rather than adding duplicates.

`POST /search` runs retrieval with no chat model involved, so you can see exactly what
retrieval returns before an agent ever reads it. (Step 07 adds a reranking stage to it.)

Try it with the three synthetic PDFs in `samples/`:

```bash
curl -X POST localhost:8000/documents -F "file=@samples/paddy_cultivation_guide.pdf" -F "category=cultivation_guide"
curl -X POST localhost:8000/documents -F "file=@samples/fertiliser_subsidy_circular_2026.pdf" -F "category=policy"
curl -X POST localhost:8000/documents -F "file=@samples/minneriya_field_records_2026.pdf" -F "category=farm_record"
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "subsidy claim deadline", "category": "policy"}'
```

On Windows, type `curl.exe` instead of `curl`, or use the `/docs` page.

### Step 04 · The RAG agent

`src/agrivibe/agents/fieldoracle/` is the agent layer. It has no idea HTTP exists.

- `tools.py`: `search_knowledge` is the retriever wrapped as a tool. Its docstring is the
  prompt that tells the model when to filter by category. It returns two things
  (`response_format="content_and_artifact"`): text for the model to read, and the raw
  documents, which ride along on the `ToolMessage` so the code can return the cited
  passages later.
- `schemas.py`: `CitedAnswer` is a list of `CitedSentence` (a sentence plus the ids of the
  passages that support it) and a `found` flag for questions the documents do not cover.
- `prompts.py`: search first, answer only from passages, cite every passage each sentence
  uses (one sentence can combine several), say so when nothing covers the question.
- `agent.py`: `create_agent(..., response_format=CitedAnswer)` joins them.

### Step 05 · The ask endpoint

`POST /ask` calls `AnswerService.ask`, which invokes the agent and builds the response.
`answer` is the sentences joined, each followed by the ids of the passages it cites:
`Sentence [id-1, id-2]`. `sentences` has the same content as a list, and `sources` holds
the full text of every cited passage, taken from what the search tool returned.

The agent has a checkpointer (`InMemorySaver`), so a conversation can go on for several
turns. Every response carries a `thread_id`. Send it back with the next question and the
agent sees the earlier turns; leave it out and a new thread starts. `sources` and
`searches` only cover the latest turn. Threads live in process memory, so they are lost
when the app restarts.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "When is the first top dressing due on the Minneriya field, and can the farmer still claim the subsidy?"}'

# a follow-up in the same conversation
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Where does the farmer apply?", "thread_id": "<thread_id from the first answer>"}'
```

Your output will differ in wording. The shape is fixed:

```json
{
  "question": "...",
  "thread_id": "3f1c9a52-...",
  "found": true,
  "answer": "The field was sown with Bg 300 on 2 October 2026. [minneriya-field-records-2026-41c1b4-0] ...",
  "sentences": [
    {"sentence": "The field was sown with Bg 300 on 2 October 2026.", "citations": ["minneriya-field-records-2026-41c1b4-0"]}
  ],
  "sources": [
    {"id": "minneriya-field-records-2026-41c1b4-0", "source": "minneriya_field_records_2026.pdf", "page": 1, "category": "farm_record", "text": "...", "rerank_score": 0.94}
  ],
  "searches": ["Minneriya field sowing date variety", "top dressing timing three-month variety", "fertiliser subsidy eligibility deadline"]
}
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | `{"status": "ok"}` |
| POST | `/documents` | form: `file` (PDF), `category` | `IndexResponse` |
| POST | `/search` | `{"query", "category"?, "k"?, "rerank"?}` | `SearchResponse` |
| POST | `/ask` | `{"question", "thread_id"?}` | `AskResponse` |

The request and response models are in `src/agrivibe/models/`, one module per router. That
package is the contract a frontend codes against.

## How the code is layered

```
src/agrivibe/
├── main.py            the FastAPI app: lifespan, middleware and routers
├── config.py          settings from .env
├── dependencies.py    builds each shared object once; tests swap them for fakes
├── models/            request and response bodies, one module per router
│   ├── knowledge.py   IndexResponse, SearchRequest, SearchResponse, Source
│   └── ask.py         AskRequest, AskResponse
├── api/               HTTP layer: routes only, no logic
│   ├── knowledge.py   POST /documents, POST /search
│   └── ask.py         POST /ask
├── service/           one class per use case
│   ├── knowledge.py   read PDF, split, tag, upsert; search
│   └── answering.py   invoke the agent, format the cited answer, collect sources
├── agents/            agent layer: knows nothing about HTTP
│   ├── fieldoracle/   the RAG agent: tool, prompt, output schema
│   ├── fieldsat.py    what the FieldSat agents share: FieldContext, FieldReport, FieldSatReport
│   ├── weather/       rain for one field, from FieldSat
│   └── farm_context/  satellite field health for one field, from FieldSat
├── core/              clients shared by every layer
│   ├── llm.py         the chat model
│   ├── reranker.py    the cross-encoder that re-scores search hits
│   └── mcp.py         the FieldSat tools, loaded over MCP
└── db/                data access: where the indexed chunks live
    └── vectorstore.py Pinecone, the Category type, search_chunks (search, then rerank)
```

Calls only go down: `api → service → agents → core, db`. A route never touches Pinecone and
an agent never sees a request.

### Step 06 · Tests without keys

`tests/fakes.py` swaps Pinecone for an in-memory store and OpenAI for a
`ScriptedChatModel` that replies with a fixed list of messages, tool calls included. The
tests replace the real objects through `app.dependency_overrides`, so the real routes,
service layer and agent graph run with no network and no cost.

```bash
uv run pytest
```

### Step 07 · Two-stage retrieval

A vector search embeds the question and each chunk separately, then compares the two
vectors. That is fast, because every chunk was embedded once at upload time, but the
question and the chunk never meet, so a chunk that shares the topic can outrank the one
that holds the answer.

A **cross-encoder** reads the question and the chunk together, as one input, and outputs
one relevance score. It sees which words of the question the chunk actually answers, so it
ranks better. The price is that it has to run once per chunk for every question; nothing
can be computed ahead of time. Running it over the whole index would take far too long.

So retrieval now runs in two stages:

1. **Recall.** Pinecone returns the `RERANK_CANDIDATES` closest chunks (default 20).
2. **Precision.** The cross-encoder scores those 20 against the question and the best
   `SEARCH_K` (default 4) are kept, best first.

Both stages live in `search_chunks` (`src/agrivibe/db/vectorstore.py`), so `POST /search`
and the agent's `search_knowledge` tool share them. The cross-encoder itself is
`Reranker` in `src/agrivibe/core/reranker.py`, built once in `dependencies.py` and passed
to `KnowledgeService` and the agent. Each hit keeps its vector `score` and gains a
`rerank_score`; each source in an `/ask` answer carries its `rerank_score` too.

Pick the cross-encoder with `RERANKER` in `.env`:

| `RERANKER` | Model | Runs | Notes |
|---|---|---|---|
| `pinecone` (default) | `RERANK_MODEL`, default `bge-reranker-v2-m3` | Pinecone's servers | Reuses `PINECONE_API_KEY`, installs nothing. The Starter plan has a monthly rerank quota; if a rerank fails, the log says so and the vector order is kept. |
| `none` | | | The Step 06 behaviour: one vector search for `SEARCH_K` chunks. |

Every rerank logs its cost, so you can see what the second stage adds to each search:

```
INFO:     agrivibe.core.reranker: Reranked 20 chunks with pinecone in <n> ms
```

To compare the two orders, send the same query with `"rerank": false`:

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "subsidy claim deadline"}'
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "subsidy claim deadline", "rerank": false}'
```

Chunks are always identified by their record id, `<doc_id>-<n>`. The `chunk_id` in the
metadata is only the chunk's position in its PDF, so every PDF has a chunk 0; it is never
used to tell chunks apart or to cite them.

The tests swap the real cross-encoder for `FakeReranker` in `tests/fakes.py`, which scores
a chunk by how many words of the query it contains.

### Step 08 · FieldSat tools over MCP

The Weather and Farm Context agents read field evidence from FieldSat, the MCP server you
built in Level 6 Lesson 3. This app never imports FieldSat's code. `core/mcp.py` asks the
server for its tools with `MCPAdapter` (from `langchain[mcp]`) and gets back ordinary
LangChain tools, with the names, descriptions and argument schemas the server published.
`pick_tools` gives each agent only the tools it needs.

Asking for the tools is async, so it happens in the app's `lifespan` in `main.py`, once at
startup. FieldSat is required in the same way as the API keys: if the server cannot be
reached, the app refuses to start and tells you how to start it. With FieldSat running,
`uv run fastapi dev` prints:

```
FieldSat tools: list_fields, get_satellite_analysis, get_weather_context
```

FieldSat listens on port 8001 because this app uses 8000. If your server runs elsewhere, set
`FIELDSAT_URL` in `.env`.

### Step 09 · The Weather Agent

`agents/weather/` has the same shape as `agents/fieldoracle/`: a prompt and one build
function. `build_weather_agent` takes the chat model and every FieldSat tool, and keeps
one: `get_weather_context`.

The agent answers with a `FieldReport` under `structured_response`, with three parts:

- `field_id`: the field it was asked about.
- `summary`: two to four sentences in the agent's own words.
- `data`: FieldSat's reply, exactly as the server sent it. `None` if the tool failed.

The model writes only the summary. It never picks the field and never copies a number,
so it cannot get either wrong. `agents/fieldsat.py` holds the pieces that make this work:

- `FieldContext(field_id)`: the field, passed with each run as runtime context:
  `ainvoke({"messages": [...]}, context=FieldContext("SL-PADDY-MINNERIYA"))`.
- `FieldSatReport`: middleware with two hooks. `awrap_tool_call` runs before every tool
  call and replaces the `field_id` the model wrote with the one from the context.
  `after_agent` runs once at the end and builds the `FieldReport` from the context, the
  agent's last message and FieldSat's last reply.

```json
{
  "field_id": "SL-PADDY-MINNERIYA",
  "summary": "112 mm of rain fell on 5 of the last 7 days, 38 mm of it in the last 48 hours. Another 46 mm is forecast for the next 48 hours, so heavy rain is expected.",
  "data": {
    "rainfall_last_7_days_mm": 112.0,
    "rainy_days_last_7_days": 5,
    "rainfall_last_48_hours_mm": 38.0,
    "rainfall_forecast_next_48_hours_mm": 46.0,
    "average_humidity_last_7_days": 88.0,
    "heavy_rain_recent": true,
    "heavy_rain_expected": true
  }
}
```

MCP tools are async, so the agent is run with `ainvoke`. Try it before any endpoint uses it:

```bash
uv run python scripts/try_agent.py weather SL-PADDY-MINNERIYA "How much rain has the field had this week?"
```

The script prints each tool call the agent made, what FieldSat answered, and the report as
JSON.

### Step 10 · The Farm Context Agent

`agents/farm_context/` repeats the pattern with `get_satellite_analysis`, and the same
`FieldSatReport` middleware. Its prompt fixes the comparison window at 14 days unless the
request names another, puts poor data quality first, and forbids naming a cause: NDVI says
the crop got less green, not why. Its `data` is FieldSat's satellite analysis: NDVI now and
before, the change and its `trend`, `field_uniformity`, `valid_observations`, cloud cover and
`data_quality`.

```bash
uv run python scripts/try_agent.py farm_context SL-PADDY-MINNERIYA "Is the field greener than two weeks ago?"
```

Two agents, one tool each, from the same server. Neither knows the other exists. Finding
the field from a place name such as Minneriya is the orchestrator's job, in step 11.

## Using it from the AgriVibe frontend

The AgriVibe Lite frontend (Next.js) keeps all its knowledge in four Markdown files
pasted into one agent's prompt. FieldOracle replaces that with retrieval:

- Its Crop Knowledge agent calls `POST /ask` (or `POST /search`) instead of reading
  `KNOWLEDGE_BASE`.
- `answer` can be shown as is. For citation chips, render `sentences` instead: one chip
  per id in `citations`.
- Keep `thread_id` per chat and send it with each follow-up.
- `sources` render as an evidence panel: file, page, passage.
- `searches` render as live tool chips, the same way satellite and weather calls show today.
- Add the frontend's URL to `CORS_ORIGINS` if the browser calls FieldOracle directly.

## Growing into a multi-agent backend

AgriVibe runs several agents in a fixed order (intake, field evidence, knowledge,
advisory). That fits this layout without moving anything:

```
src/agrivibe/
├── agents/
│   ├── fieldoracle/   ← exists
│   ├── intake/        new: build_intake_agent()
│   └── advisory/      new: build_advisory_agent()
├── workflows/
│   └── advisory.py    new: runs the agents in order, passing typed state between them
├── service/
│   └── advising.py    new: AdvisoryService invokes the workflow
└── api/
    └── advise.py      new: POST /advise
```

Each agent stays a package with one build function. The RAG agent can be handed to the
others as a tool, and `/ask` keeps working on its own.

