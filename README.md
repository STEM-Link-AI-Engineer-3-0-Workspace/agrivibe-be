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

<!-- step-02 -->
<!-- step-03 -->
<!-- step-04 -->
<!-- step-05 -->
<!-- step-06 -->
