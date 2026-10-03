# FieldOracle

The knowledge service behind AgriVibe. You upload PDFs (cultivation guides, government
circulars, farm records) and FieldOracle splits them, embeds them and stores them in
Pinecone. You ask a question and a RAG agent searches that store and answers in
sentence and citation pairs, so every sentence traces back to a passage in a PDF.

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

`POST /search` runs a plain vector search with no model involved, so you can see exactly
what retrieval returns before an agent ever reads it.

Try it with the three synthetic PDFs in `samples/`:

```bash
curl -X POST localhost:8000/documents -F "file=@samples/paddy_cultivation_guide.pdf" -F "category=cultivation_guide"
curl -X POST localhost:8000/documents -F "file=@samples/fertiliser_subsidy_circular_2026.pdf" -F "category=policy"
curl -X POST localhost:8000/documents -F "file=@samples/minneriya_field_records_2026.pdf" -F "category=farm_record"
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "subsidy claim deadline", "category": "policy"}'
```

On Windows, type `curl.exe` instead of `curl`, or use the `/docs` page.

### Step 04 · The RAG agent

`src/agrivibe/agents/rag/` is the agent layer. It has no idea HTTP exists.

- `tools.py`: `search_knowledge` is the retriever wrapped as a tool. Its docstring is the
  prompt that tells the model when to filter by category. It returns two things
  (`response_format="content_and_artifact"`): text for the model to read, and the raw
  documents, which ride along on the `ToolMessage` so the code can check citations later.
- `schemas.py`: `CitedAnswer` is a list of `CitedSentence` (sentence plus citation) and a
  `found` flag for questions the documents do not cover.
- `prompts.py`: search first, answer only from passages, cite every sentence, say so when
  nothing covers the question.
- `agent.py`: `create_agent(..., response_format=CitedAnswer)` joins them.

### Step 05 · The ask endpoint

`POST /ask` calls `AnswerService.ask`, which invokes the agent and builds the response.
Before it answers, it runs a grounding check. Every citation is compared with the chunks the
search tool actually returned in this run, and a citation to a passage that was never
retrieved is marked `verified: false`.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "When is the first top dressing due on the Minneriya field, and can the farmer still claim the subsidy?"}'
```

Your output will differ in wording. The shape is fixed:

```json
{
  "question": "...",
  "found": true,
  "answer": "The field was sown with Bg 300 on 2 October 2026. ...",
  "sentences": [
    {"sentence": "The field was sown with Bg 300 on 2 October 2026.", "citation": "minneriya-field-records-2026-41c1b4-0", "verified": true}
  ],
  "sources": [
    {"id": "minneriya-field-records-2026-41c1b4-0", "source": "minneriya_field_records_2026.pdf", "page": 1, "category": "farm_record", "text": "..."}
  ],
  "searches": ["Minneriya field sowing date variety", "top dressing timing three-month variety", "fertiliser subsidy eligibility deadline"]
}
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | `{"status": "ok"}` |
| POST | `/documents` | form: `file` (PDF), `category` | `IndexResponse` |
| POST | `/search` | `{"query", "category"?, "k"?}` | `SearchResponse` |
| POST | `/ask` | `{"question"}` | `AskResponse` |

The request and response models are in `src/agrivibe/models/`, one module per router. That
package is the contract a frontend codes against.

## How the code is layered

```
src/agrivibe/
├── main.py            the FastAPI app: middleware and routers
├── config.py          settings from .env
├── dependencies.py    builds each shared object once; tests swap them for fakes
├── models/            request and response bodies, one module per router
│   ├── knowledge.py   IndexResponse, SearchRequest, SearchResponse, Source
│   └── ask.py         AskRequest, AskResponse
├── api/               HTTP layer: routes only, no logic
│   ├── knowledge.py   POST /documents, POST /search
│   └── ask.py         POST /ask
├── services/          one class per use case
│   ├── knowledge.py   read PDF, split, tag, upsert; search
│   └── answering.py   invoke the agent, check citations, shape the response
├── agents/            agent layer: knows nothing about HTTP
│   └── rag/           the RAG agent: tool, prompt, output schema
├── core/              model clients shared by every layer
│   └── llm.py         the chat model
└── db/                data access: where the indexed chunks live
    └── vectorstore.py Pinecone, the Category type, search_chunks
```

Calls only go down: `api → services → agents → core, db`. A route never touches Pinecone and
an agent never sees a request.

### Step 06 · Tests without keys

`tests/fakes.py` swaps Pinecone for an in-memory store and OpenAI for a
`ScriptedChatModel` that replies with a fixed list of messages, tool calls included. The
tests replace the real objects through `app.dependency_overrides`, so the real routes,
services and agent graph run with no network and no cost.

```bash
uv run pytest
```

## Using it from the AgriVibe frontend

The AgriVibe Lite frontend (Next.js) keeps all its knowledge in four Markdown files
pasted into one agent's prompt. FieldOracle replaces that with retrieval:

- Its Crop Knowledge agent calls `POST /ask` (or `POST /search`) instead of reading
  `KNOWLEDGE_BASE`.
- `sentences` render as the answer, one citation chip per sentence. A sentence with
  `verified: false` should look different.
- `sources` render as an evidence panel: file, page, passage.
- `searches` render as live tool chips, the same way satellite and weather calls show today.
- Add the frontend's URL to `CORS_ORIGINS` if the browser calls FieldOracle directly.

## Growing into a multi-agent backend

AgriVibe runs several agents in a fixed order (intake, field evidence, knowledge,
advisory). That fits this layout without moving anything:

```
src/agrivibe/
├── agents/
│   ├── rag/           ← exists
│   ├── intake/        new: build_intake_agent()
│   └── advisory/      new: build_advisory_agent()
├── workflows/
│   └── advisory.py    new: runs the agents in order, passing typed state between them
├── services/
│   └── advising.py    new: AdvisoryService invokes the workflow
└── api/
    └── advise.py      new: POST /advise
```

Each agent stays a package with one build function. The RAG agent can be handed to the
others as a tool, and `/ask` keeps working on its own.

