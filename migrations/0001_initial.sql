-- Context Guard schema. Timestamps are RFC 3339 UTC text.

CREATE TABLE conversations (
    id                   TEXT PRIMARY KEY,
    id_source            TEXT NOT NULL,
    user_id              TEXT,
    model                TEXT NOT NULL,
    first_seen           TEXT NOT NULL,
    last_seen            TEXT NOT NULL,
    turns                INTEGER NOT NULL DEFAULT 0,
    last_health          INTEGER,
    last_risk            INTEGER,
    last_status          TEXT,
    last_context_percent REAL,
    last_prompt_tokens   INTEGER,
    last_context_limit   INTEGER,
    last_messages_count  INTEGER NOT NULL DEFAULT 0,
    last_messages_hash   TEXT
);
CREATE INDEX idx_conversations_last_seen ON conversations(last_seen);

CREATE TABLE conversation_events (
    id                 TEXT PRIMARY KEY,
    conversation_id    TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn               INTEGER,
    kind               TEXT NOT NULL,
    ts                 TEXT NOT NULL,
    model              TEXT NOT NULL,
    message_id         TEXT,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    context_limit      INTEGER,
    response_text      TEXT,
    new_messages_json  TEXT,
    full_messages_json TEXT
);
CREATE INDEX idx_events_conv_turn ON conversation_events(conversation_id, turn);

CREATE TABLE health_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn            INTEGER NOT NULL,
    ts              TEXT NOT NULL,
    message_id      TEXT,
    health          INTEGER NOT NULL,
    risk            INTEGER NOT NULL,
    status          TEXT NOT NULL,
    context_percent REAL,
    prompt_tokens   INTEGER,
    context_limit   INTEGER,
    reasons_json    TEXT NOT NULL,
    summary         TEXT NOT NULL
);
CREATE INDEX idx_health_conv_turn ON health_results(conversation_id, turn);
CREATE INDEX idx_health_conv_msg ON health_results(conversation_id, message_id);

CREATE TABLE known_values (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    anchor          TEXT NOT NULL,
    value           TEXT NOT NULL,
    source_role     TEXT NOT NULL,
    first_turn      INTEGER NOT NULL,
    last_turn       INTEGER NOT NULL,
    occurrences     INTEGER NOT NULL DEFAULT 1,
    UNIQUE(conversation_id, kind, anchor, value)
);

CREATE TABLE identifiers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    value           TEXT NOT NULL,
    source_role     TEXT NOT NULL,
    first_turn      INTEGER NOT NULL,
    UNIQUE(conversation_id, kind, value)
);

CREATE TABLE tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    call_id         TEXT NOT NULL,
    turn            INTEGER NOT NULL,
    name            TEXT NOT NULL,
    args_hash       TEXT NOT NULL,
    args_json       TEXT,
    result_turn     INTEGER,
    result_hash     TEXT,
    result_status   TEXT,
    ts              TEXT NOT NULL
);
CREATE INDEX idx_tool_calls_conv ON tool_calls(conversation_id, id);
CREATE INDEX idx_tool_calls_call_id ON tool_calls(conversation_id, call_id);

CREATE TABLE anomalies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn            INTEGER NOT NULL,
    ts              TEXT NOT NULL,
    signal          TEXT NOT NULL,
    penalty         INTEGER NOT NULL,
    severity        TEXT NOT NULL,
    detail          TEXT NOT NULL,
    dedupe_key      TEXT NOT NULL,
    UNIQUE(conversation_id, dedupe_key)
);
CREATE INDEX idx_anomalies_conv_turn ON anomalies(conversation_id, turn);
