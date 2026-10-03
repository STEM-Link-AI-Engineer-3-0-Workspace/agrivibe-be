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

`src/agrivibe/core/vectorstore.py` connects to Pinecone and creates the index on first run with
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

<!-- step-04 -->
<!-- step-05 -->
<!-- step-06 -->
