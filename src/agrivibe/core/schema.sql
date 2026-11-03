-- AgriVibe's own tables: the farmer, their conversations, and every turn in them.
--
--     farmers ──< threads ──< turns
--
-- Applied on every start (open_database runs it), so each statement must be safe to repeat.
-- LangGraph's tables (checkpoints, checkpoint_blobs, checkpoint_writes, store) sit beside
-- these and are created by LangGraph itself.

CREATE TABLE IF NOT EXISTS farmers (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS threads (
    id          uuid PRIMARY KEY,
    farmer_id   text NOT NULL REFERENCES farmers (id) ON DELETE CASCADE,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- The sidebar's query: one farmer's threads, most recently used first.
CREATE INDEX IF NOT EXISTS threads_by_farmer ON threads (farmer_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id     uuid NOT NULL REFERENCES threads (id) ON DELETE CASCADE,
    position      integer NOT NULL,           -- 0, 1, 2… within the thread
    message       text NOT NULL,
    answer        text NOT NULL,
    tools_called  text[] NOT NULL,
    requirements  jsonb,                      -- what the orchestrator saved: field_id, crop, problem, timeframe
    findings      jsonb NOT NULL,             -- each specialist's report, sources included, as the farmer saw it
    started_at    timestamptz NOT NULL,
    seconds       double precision NOT NULL,
    UNIQUE (thread_id, position)              -- also the index that reads a thread's turns in order
);
