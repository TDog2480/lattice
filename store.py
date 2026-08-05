"""SQLite store for normalized Claude Code transcript events.

Schema notes, grounded in Phase 1 discovery (discovery_report.md):

- Token usage in the source is PER-CALL, not cumulative (verified: input_tokens
  is ~2 with the context in cache fields; output_tokens non-monotonic). So we
  store values as recorded; no cumulative->delta normalization is needed.
- The same API call is logged as multiple assistant entries (one per content
  block) sharing a requestId with identical usage. The `usage` table holds ONE
  row per (session_id, request_id); `n_log_entries` records how many raw lines
  were collapsed, as evidence the dedupe happened.
- Tool result token counts are NOT in the source. `tool_results.est_tokens` is
  chars/4 — an ESTIMATE, and every consumer must label it as such.
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_files(
  path          TEXT PRIMARY KEY,
  byte_offset   INTEGER NOT NULL,
  line_no       INTEGER NOT NULL,
  size          INTEGER,
  mtime         REAL,
  malformed     INTEGER DEFAULT 0,
  last_ingested TEXT
);

CREATE TABLE IF NOT EXISTS sessions(
  id           TEXT PRIMARY KEY,
  project      TEXT,
  cwd          TEXT,
  git_branch   TEXT,
  entrypoint   TEXT,
  versions     TEXT,   -- distinct client versions seen, comma-joined
  models       TEXT,   -- distinct models seen, comma-joined
  first_ts     TEXT,
  last_ts      TEXT,
  n_entries    INTEGER,
  n_api_calls  INTEGER,
  n_tool_calls INTEGER
);

-- conversational entries only (those carrying a uuid)
CREATE TABLE IF NOT EXISTS turns(
  uuid         TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  line_no      INTEGER NOT NULL,
  type         TEXT,
  subtype      TEXT,
  role         TEXT,
  parent_uuid  TEXT,
  ts           TEXT,
  is_sidechain INTEGER,
  is_meta      INTEGER,
  is_human     INTEGER,  -- user entry that is a prompt, not a tool result
  is_exploratory INTEGER, -- human prompt matched exploratory keywords (flag only; text never stored)
  request_id   TEXT,
  model        TEXT
);

-- one row per API call (deduped by request_id; see module docstring)
CREATE TABLE IF NOT EXISTS usage(
  session_id       TEXT NOT NULL,
  request_id       TEXT NOT NULL,   -- 'uuid:<uuid>' when source lacks requestId
  line_no          INTEGER,
  ts               TEXT,
  model            TEXT,
  input_tokens     INTEGER,
  output_tokens    INTEGER,
  cache_creation   INTEGER,
  cache_read       INTEGER,
  eph_1h           INTEGER,
  eph_5m           INTEGER,
  n_iterations     INTEGER,         -- >1 means server-side retry
  cache_miss_type  TEXT,            -- from message.diagnostics.cache_miss_reason
  cache_miss_tokens INTEGER,
  n_log_entries    INTEGER,
  PRIMARY KEY(session_id, request_id)
);

CREATE TABLE IF NOT EXISTS tool_calls(
  id           TEXT PRIMARY KEY,    -- tool_use block id
  session_id   TEXT NOT NULL,
  line_no      INTEGER,
  entry_uuid   TEXT,
  ts           TEXT,
  tool_name    TEXT,
  args_digest  TEXT,                -- sha256 of canonical input JSON
  target_path  TEXT,                -- file_path/path/notebook_path if present
  command_head TEXT,                -- first token of Bash/PowerShell command
  read_offset  INTEGER,
  read_limit   INTEGER,
  input_chars  INTEGER
);

CREATE TABLE IF NOT EXISTS tool_results(
  tool_use_id  TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  line_no      INTEGER,
  ts           TEXT,
  is_error     INTEGER,
  content_chars INTEGER,
  est_tokens   INTEGER,             -- chars/4 ESTIMATE, not measured
  result_type  TEXT,                -- toolUseResult.type when present
  stdout_chars INTEGER,
  stderr_chars INTEGER,
  interrupted  INTEGER
);

CREATE TABLE IF NOT EXISTS file_events(
  id           INTEGER PRIMARY KEY,
  session_id   TEXT NOT NULL,
  line_no      INTEGER,
  ts           TEXT,
  tool_use_id  TEXT,
  path         TEXT,
  operation    TEXT,                -- read | edit | write
  read_offset  INTEGER,
  read_limit   INTEGER,
  num_lines    INTEGER,             -- lines returned (read)
  start_line   INTEGER,
  total_lines  INTEGER,             -- file's total lines at read time
  content_chars INTEGER,
  content_digest TEXT,              -- sha256 of returned content; content itself never stored
  patch_hunks  INTEGER,             -- edits: hunk count
  patch_lines  INTEGER,             -- edits: lines touched (max(old,new) summed)
  user_modified INTEGER,
  UNIQUE(session_id, line_no, operation, path)
);

-- api errors, tools_delta, mode changes, file-history deltas, and any future
-- compaction/subagent markers land here
CREATE TABLE IF NOT EXISTS events(
  id         INTEGER PRIMARY KEY,
  session_id TEXT,
  line_no    INTEGER,
  ts         TEXT,
  kind       TEXT,
  detail     TEXT,                  -- small structural JSON, never content
  UNIQUE(session_id, line_no, kind)
);

CREATE TABLE IF NOT EXISTS tool_availability(
  session_id TEXT,
  tool_name  TEXT,
  action     TEXT,                  -- added | removed | readded
  line_no    INTEGER,
  UNIQUE(session_id, tool_name, action, line_no)
);

CREATE TABLE IF NOT EXISTS schema_warnings(
  kind       TEXT,                  -- top_key | entry_type | usage_field | content_block
  name       TEXT,
  count      INTEGER,
  first_file TEXT,
  first_line INTEGER,
  UNIQUE(kind, name)
);

CREATE INDEX IF NOT EXISTS idx_turns_session   ON turns(session_id, line_no);
CREATE INDEX IF NOT EXISTS idx_usage_session   ON usage(session_id, line_no);
CREATE INDEX IF NOT EXISTS idx_calls_session   ON tool_calls(session_id, line_no);
CREATE INDEX IF NOT EXISTS idx_fev_session     ON file_events(session_id, line_no);
CREATE INDEX IF NOT EXISTS idx_fev_path        ON file_events(path);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id, line_no);
"""

SESSION_TABLES = (
    "turns", "usage", "tool_calls", "tool_results",
    "file_events", "events", "tool_availability",
)


def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def purge_session(con: sqlite3.Connection, session_id: str) -> None:
    """Remove all derived rows for a session (used when its file was rewritten)."""
    for t in SESSION_TABLES:
        con.execute(f"DELETE FROM {t} WHERE session_id = ?", (session_id,))
    con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def bump_warning(con: sqlite3.Connection, kind: str, name: str,
                 file: str, line: int) -> None:
    con.execute(
        """INSERT INTO schema_warnings(kind, name, count, first_file, first_line)
           VALUES(?,?,1,?,?)
           ON CONFLICT(kind, name) DO UPDATE SET count = count + 1""",
        (kind, name, file, line))
