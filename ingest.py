"""Phase 2: incremental ingest of Claude Code JSONL transcripts into SQLite.

Usage:
    python ingest.py [--db lattice.db] [--projects-dir PATH] [--rebuild] [-q]

Guarantees:
- Source files are opened read-only ('rb') and never modified.
- Incremental: a byte-offset cursor per file; re-running never duplicates rows.
  A file that shrank or was rewritten is purged and re-ingested whole.
- Unknown top-level keys, entry types, usage fields, and content-block types
  are recorded in schema_warnings and reported — never a crash.
- Malformed lines are counted per file and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from store import bump_warning, open_db, purge_session

# -- known schema (from Phase 1 discovery; unknowns are warned, not dropped) --

KNOWN_TOP_KEYS = {
    "type", "sessionId", "timestamp", "parentUuid", "isSidechain", "uuid",
    "userType", "entrypoint", "cwd", "version", "gitBranch", "message",
    "requestId", "promptId", "effort", "toolUseResult",
    "sourceToolAssistantUUID", "operation", "aiTitle", "leafUuid",
    "lastPrompt", "slug", "permissionMode", "promptSource", "attachment",
    "messageId", "snapshot", "isSnapshotUpdate", "subtype", "level", "error",
    "retryInMs", "retryAttempt", "maxRetries", "origin", "source", "content",
    "attributionMcpServer", "attributionMcpTool", "mode", "attributionSkill",
    "snapshotMessageId", "trackingPath", "backup", "isMeta",
    "isApiErrorMessage", "toolDenialKind", "sourceToolUseID",
}

KNOWN_ENTRY_TYPES = {
    "assistant", "user", "queue-operation", "ai-title", "last-prompt",
    "attachment", "file-history-snapshot", "system", "mode",
    "file-history-delta", "summary",
}

KNOWN_USAGE_FIELDS = {
    "input_tokens", "output_tokens", "cache_creation_input_tokens",
    "cache_read_input_tokens", "cache_creation", "server_tool_use",
    "service_tier", "inference_geo", "iterations", "speed",
}

KNOWN_CONTENT_BLOCKS = {"text", "thinking", "tool_use", "tool_result", "image"}

FILE_TOOL_OPS = {"Read": "read", "Edit": "edit", "Write": "write",
                 "MultiEdit": "edit", "NotebookEdit": "edit"}

# D6 exclusion: prompts that are explicitly exploratory (searching is the task,
# not thrash). Matched at ingest; only the boolean is stored, never the text.
EXPLORATORY_RE = re.compile(
    r"\b(explore|find all|audit|overview|what does th|how does th|"
    r"look (around|through)|familiari[sz]e|walk me through)\b", re.IGNORECASE)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]


def content_chars(content) -> int:
    """Char length of a tool_result content payload (str or list of blocks)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for b in content:
            if isinstance(b, dict):
                n += len(b.get("text", "") or "")
            elif isinstance(b, str):
                n += len(b)
        return n
    return 0


class FileIngestor:
    """Ingests one transcript file from a byte-offset cursor onward."""

    def __init__(self, con, path: Path, project: str):
        self.con = con
        self.path = path
        self.project = project
        self.session_id = path.stem
        self.fname = path.name
        self.malformed = 0
        self.entries = 0
        # request_key -> usage row dict (deduped within this run)
        self.usage_buf: dict[str, dict] = {}
        # tool_use id -> {tool_name, read_offset, read_limit} for result joining
        self.pending_calls: dict[str, dict] = {}
        self.meta = {"cwd": None, "git_branch": None, "entrypoint": None,
                     "versions": set(), "models": set()}

    # -- per-entry handlers -------------------------------------------------

    def handle(self, e: dict, line_no: int) -> None:
        self.entries += 1
        etype = e.get("type", "<missing>")
        for k in e:
            if k not in KNOWN_TOP_KEYS:
                bump_warning(self.con, "top_key", k, self.fname, line_no)
        if etype not in KNOWN_ENTRY_TYPES:
            bump_warning(self.con, "entry_type", etype, self.fname, line_no)

        m = self.meta
        m["cwd"] = m["cwd"] or e.get("cwd")
        m["git_branch"] = m["git_branch"] or e.get("gitBranch")
        m["entrypoint"] = m["entrypoint"] or e.get("entrypoint")
        if isinstance(e.get("version"), str):
            m["versions"].add(e["version"])

        msg = e.get("message") if isinstance(e.get("message"), dict) else None
        if msg and isinstance(msg.get("model"), str):
            m["models"].add(msg["model"])

        is_human = is_exploratory = 0
        if etype == "user" and msg and not e.get("isMeta"):
            content = msg.get("content")
            blocks = content if isinstance(content, list) else []
            has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result"
                                  for b in blocks)
            if not has_tool_result:
                is_human = 1
                texts = [content] if isinstance(content, str) else \
                    [b.get("text", "") for b in blocks
                     if isinstance(b, dict) and b.get("type") == "text"]
                if EXPLORATORY_RE.search(" ".join(texts)[:4000]):
                    is_exploratory = 1

        if isinstance(e.get("uuid"), str):
            self.con.execute(
                """INSERT OR REPLACE INTO turns
                   (uuid, session_id, line_no, type, subtype, role, parent_uuid,
                    ts, is_sidechain, is_meta, is_human, is_exploratory,
                    request_id, model)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (e["uuid"], self.session_id, line_no, etype, e.get("subtype"),
                 msg.get("role") if msg else None, e.get("parentUuid"),
                 e.get("timestamp"), int(bool(e.get("isSidechain"))),
                 int(bool(e.get("isMeta"))), is_human, is_exploratory,
                 e.get("requestId"),
                 msg.get("model") if msg else None))

        if etype == "assistant" and msg:
            self._usage(e, msg, line_no)
        if msg and isinstance(msg.get("content"), list):
            self._content_blocks(e, msg, line_no)

        if etype == "system":
            self._event(line_no, e.get("timestamp"), "api_error"
                        if e.get("subtype") == "api_error" else f"system:{e.get('subtype')}",
                        {"subtype": e.get("subtype"), "level": e.get("level"),
                         "source": e.get("source"),
                         "retryAttempt": e.get("retryAttempt")})
        elif etype == "attachment":
            att = e.get("attachment") or {}
            if att.get("type") == "deferred_tools_delta":
                for action, key in (("added", "addedNames"),
                                    ("removed", "removedNames"),
                                    ("readded", "readdedNames")):
                    for name in att.get(key) or []:
                        self.con.execute(
                            """INSERT OR IGNORE INTO tool_availability
                               (session_id, tool_name, action, line_no)
                               VALUES(?,?,?,?)""",
                            (self.session_id, name, action, line_no))
            else:
                self._event(line_no, e.get("timestamp"),
                            f"attachment:{att.get('type')}", {})
        elif etype == "file-history-delta":
            self._event(line_no, e.get("timestamp"), "file_history_delta",
                        {"trackingPath": e.get("trackingPath"),
                         "version": (e.get("backup") or {}).get("version")})
        elif etype == "mode":
            self._event(line_no, None, "mode", {"mode": e.get("mode")})
        # queue-operation / ai-title / last-prompt / file-history-snapshot:
        # bookkeeping with no token or file signal; deliberately not stored.

    def _event(self, line_no, ts, kind, detail: dict) -> None:
        detail = {k: v for k, v in detail.items() if v is not None}
        self.con.execute(
            """INSERT OR IGNORE INTO events(session_id, line_no, ts, kind, detail)
               VALUES(?,?,?,?,?)""",
            (self.session_id, line_no, ts, kind, json.dumps(detail)))

    def _usage(self, e: dict, msg: dict, line_no: int) -> None:
        u = msg.get("usage")
        if not isinstance(u, dict):
            return
        for k in u:
            if k not in KNOWN_USAGE_FIELDS:
                bump_warning(self.con, "usage_field", k, self.fname, line_no)
        key = e.get("requestId") or f"uuid:{e.get('uuid')}"
        cc = u.get("cache_creation") or {}
        diag = (msg.get("diagnostics") or {}) if isinstance(msg.get("diagnostics"), dict) else {}
        miss = diag.get("cache_miss_reason") or {}
        row = self.usage_buf.get(key)
        if row is None:
            row = self.usage_buf[key] = {
                "line_no": line_no, "ts": e.get("timestamp"),
                "model": msg.get("model"),
                "input": u.get("input_tokens"), "output": u.get("output_tokens"),
                "cache_creation": u.get("cache_creation_input_tokens"),
                "cache_read": u.get("cache_read_input_tokens"),
                "eph_1h": cc.get("ephemeral_1h_input_tokens"),
                "eph_5m": cc.get("ephemeral_5m_input_tokens"),
                "n_iterations": len(u.get("iterations") or []) or 1,
                "miss_type": miss.get("type"),
                "miss_tokens": miss.get("cache_missed_input_tokens"),
                "n": 0,
            }
        # duplicates carry identical usage; keep max output defensively
        row["n"] += 1
        row["line_no"] = line_no
        if (u.get("output_tokens") or 0) > (row["output"] or 0):
            row["output"] = u.get("output_tokens")
        if miss.get("type") and not row["miss_type"]:
            row["miss_type"] = miss.get("type")
            row["miss_tokens"] = miss.get("cache_missed_input_tokens")

    def _content_blocks(self, e: dict, msg: dict, line_no: int) -> None:
        results = []
        for b in msg["content"]:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype not in KNOWN_CONTENT_BLOCKS:
                bump_warning(self.con, "content_block", str(btype), self.fname, line_no)
            if btype == "tool_use":
                self._tool_use(e, b, line_no)
            elif btype == "tool_result":
                results.append(b)
        for b in results:
            # top-level toolUseResult belongs to the entry; attach it only when
            # unambiguous (one result block per entry — the observed norm)
            tur = e.get("toolUseResult") if len(results) == 1 else None
            self._tool_result(e, b, tur if isinstance(tur, dict) else None, line_no)

    def _tool_use(self, e: dict, b: dict, line_no: int) -> None:
        inp = b.get("input") if isinstance(b.get("input"), dict) else {}
        name = b.get("name", "?")
        target = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
        cmd_head = None
        if name in ("Bash", "PowerShell") and isinstance(inp.get("command"), str):
            cmd_head = inp["command"].strip().split()[0][:60] if inp["command"].strip() else None
        inp_json = json.dumps(inp, sort_keys=True, default=str)
        self.con.execute(
            """INSERT OR REPLACE INTO tool_calls
               (id, session_id, line_no, entry_uuid, ts, tool_name, args_digest,
                target_path, command_head, read_offset, read_limit, input_chars)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (b.get("id"), self.session_id, line_no, e.get("uuid"),
             e.get("timestamp"), name, sha256(inp_json), target, cmd_head,
             inp.get("offset"), inp.get("limit"), len(inp_json)))
        self.pending_calls[b.get("id")] = {
            "tool_name": name, "read_offset": inp.get("offset"),
            "read_limit": inp.get("limit"), "target": target,
        }

    def _lookup_call(self, tool_use_id):
        call = self.pending_calls.get(tool_use_id)
        if call:
            return call
        r = self.con.execute(
            "SELECT tool_name, read_offset, read_limit, target_path FROM tool_calls WHERE id=?",
            (tool_use_id,)).fetchone()
        if r:
            return {"tool_name": r["tool_name"], "read_offset": r["read_offset"],
                    "read_limit": r["read_limit"], "target": r["target_path"]}
        return {}

    def _tool_result(self, e: dict, b: dict, tur: dict | None, line_no: int) -> None:
        tid = b.get("tool_use_id")
        chars = content_chars(b.get("content"))
        tur = tur or {}
        stdout = tur.get("stdout") if isinstance(tur.get("stdout"), str) else None
        stderr = tur.get("stderr") if isinstance(tur.get("stderr"), str) else None
        self.con.execute(
            """INSERT OR REPLACE INTO tool_results
               (tool_use_id, session_id, line_no, ts, is_error, content_chars,
                est_tokens, result_type, stdout_chars, stderr_chars, interrupted)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, self.session_id, line_no, e.get("timestamp"),
             int(bool(b.get("is_error"))), chars, chars // 4,
             tur.get("type"),
             len(stdout) if stdout is not None else None,
             len(stderr) if stderr is not None else None,
             int(bool(tur.get("interrupted"))) if "interrupted" in tur else None))
        if tur:
            self._file_event(e, tid, tur, line_no)

    def _file_event(self, e: dict, tid, tur: dict, line_no: int) -> None:
        call = self._lookup_call(tid)
        op = FILE_TOOL_OPS.get(call.get("tool_name", ""))
        f = tur.get("file") if isinstance(tur.get("file"), dict) else None
        row = None
        if f and f.get("filePath"):  # Read result
            content = f.get("content") or ""
            row = dict(path=f["filePath"], operation="read",
                       read_offset=call.get("read_offset"),
                       read_limit=call.get("read_limit"),
                       num_lines=f.get("numLines"), start_line=f.get("startLine"),
                       total_lines=f.get("totalLines"),
                       content_chars=len(content), content_digest=sha256(content),
                       patch_hunks=None, patch_lines=None, user_modified=None)
        elif tur.get("filePath") and isinstance(tur.get("structuredPatch"), list):
            hunks = [h for h in tur["structuredPatch"] if isinstance(h, dict)]
            touched = sum(max(h.get("oldLines") or 0, h.get("newLines") or 0)
                          for h in hunks)
            content = tur.get("content") or ""
            row = dict(path=tur["filePath"],
                       operation="write" if tur.get("type") == "create" else (op or "edit"),
                       read_offset=None, read_limit=None, num_lines=None,
                       start_line=None, total_lines=None,
                       content_chars=len(content) or None,
                       content_digest=sha256(content) if content else None,
                       patch_hunks=len(hunks), patch_lines=touched,
                       user_modified=int(bool(tur.get("userModified")))
                       if "userModified" in tur else None)
        if row:
            self.con.execute(
                """INSERT OR IGNORE INTO file_events
                   (session_id, line_no, ts, tool_use_id, path, operation,
                    read_offset, read_limit, num_lines, start_line, total_lines,
                    content_chars, content_digest, patch_hunks, patch_lines,
                    user_modified)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.session_id, line_no, e.get("timestamp"), tid,
                 row["path"], row["operation"], row["read_offset"],
                 row["read_limit"], row["num_lines"], row["start_line"],
                 row["total_lines"], row["content_chars"], row["content_digest"],
                 row["patch_hunks"], row["patch_lines"], row["user_modified"]))

    # -- finalization -------------------------------------------------------

    def flush(self) -> None:
        for key, r in self.usage_buf.items():
            self.con.execute(
                """INSERT INTO usage
                   (session_id, request_id, line_no, ts, model, input_tokens,
                    output_tokens, cache_creation, cache_read, eph_1h, eph_5m,
                    n_iterations, cache_miss_type, cache_miss_tokens, n_log_entries)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id, request_id) DO UPDATE SET
                     n_log_entries = usage.n_log_entries + excluded.n_log_entries,
                     line_no = excluded.line_no""",
                (self.session_id, key, r["line_no"], r["ts"], r["model"],
                 r["input"], r["output"], r["cache_creation"], r["cache_read"],
                 r["eph_1h"], r["eph_5m"], r["n_iterations"],
                 r["miss_type"], r["miss_tokens"], r["n"]))

        old = self.con.execute("SELECT versions, models FROM sessions WHERE id=?",
                               (self.session_id,)).fetchone()
        versions, models = set(self.meta["versions"]), set(self.meta["models"])
        if old:
            versions |= set((old["versions"] or "").split(",")) - {""}
            models |= set((old["models"] or "").split(",")) - {""}
        agg = self.con.execute(
            """SELECT MIN(ts) a, MAX(ts) b, COUNT(*) n FROM turns WHERE session_id=?""",
            (self.session_id,)).fetchone()
        n_api = self.con.execute("SELECT COUNT(*) n FROM usage WHERE session_id=?",
                                 (self.session_id,)).fetchone()["n"]
        n_tc = self.con.execute("SELECT COUNT(*) n FROM tool_calls WHERE session_id=?",
                                (self.session_id,)).fetchone()["n"]
        self.con.execute(
            """INSERT OR REPLACE INTO sessions
               (id, project, cwd, git_branch, entrypoint, versions, models,
                first_ts, last_ts, n_entries, n_api_calls, n_tool_calls)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session_id, self.project, self.meta["cwd"],
             self.meta["git_branch"], self.meta["entrypoint"],
             ",".join(sorted(versions)), ",".join(sorted(models)),
             agg["a"], agg["b"], agg["n"], n_api, n_tc))


def ingest_file(con, path: Path, project: str, quiet: bool) -> str:
    """Returns one of: 'skipped', 'resumed', 'full'."""
    st = path.stat()
    cur = con.execute("SELECT * FROM ingest_files WHERE path=?",
                      (str(path),)).fetchone()
    offset, line_no, malformed, mode = 0, 0, 0, "full"
    if cur:
        if st.st_size == cur["size"] and st.st_mtime == cur["mtime"]:
            return "skipped"
        if st.st_size >= cur["byte_offset"]:
            offset, line_no = cur["byte_offset"], cur["line_no"]
            malformed, mode = cur["malformed"], "resumed"
        else:  # rewritten/truncated: purge and start over
            purge_session(con, path.stem)

    ing = FileIngestor(con, path, project)
    with path.open("rb") as fh:  # read-only, always
        fh.seek(offset)
        while True:
            raw = fh.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break  # partial trailing line still being written; retry next run
            offset = fh.tell()
            line_no += 1
            s = raw.decode("utf-8", "replace").strip()
            if not s:
                continue
            try:
                e = json.loads(s)
                if not isinstance(e, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                malformed += 1
                continue
            try:
                ing.handle(e, line_no)
            except Exception as ex:  # schema drift must never kill the run
                malformed += 1
                if not quiet:
                    print(f"  warn: {path.name}:{line_no}: {type(ex).__name__}: {ex}",
                          file=sys.stderr)
    ing.malformed = malformed
    ing.flush()
    con.execute(
        """INSERT OR REPLACE INTO ingest_files
           (path, byte_offset, line_no, size, mtime, malformed, last_ingested)
           VALUES(?,?,?,?,?,?,?)""",
        (str(path), offset, line_no, st.st_size, st.st_mtime, malformed,
         datetime.now(timezone.utc).isoformat()))
    return mode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="lattice.db")
    ap.add_argument("--projects-dir", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--rebuild", action="store_true",
                    help="drop the DB file's contents and re-ingest everything")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.projects_dir)
    if not root.is_dir():
        sys.exit(f"projects dir not found: {root}")

    if args.rebuild and Path(args.db).exists():
        Path(args.db).unlink()
    con = open_db(args.db)

    counts = {"full": 0, "resumed": 0, "skipped": 0}
    files = sorted(root.rglob("*.jsonl"))
    for f in files:
        project = f.parent.name if f.parent != root else "<root>"
        mode = ingest_file(con, f, project, args.quiet)
        counts[mode] += 1
        if not args.quiet and mode != "skipped":
            print(f"  {mode:>7}  {f.parent.name}/{f.name}")
    con.commit()

    ns = con.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"]
    stats = {t: con.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"]
             for t in ("turns", "usage", "tool_calls", "tool_results",
                       "file_events", "events", "tool_availability")}
    mal = con.execute("SELECT SUM(malformed) n FROM ingest_files").fetchone()["n"] or 0
    print(f"\n{len(files)} files: {counts['full']} full, {counts['resumed']} resumed, "
          f"{counts['skipped']} unchanged. {ns} sessions, {mal} malformed lines.")
    print("  " + ", ".join(f"{t}={n}" for t, n in stats.items()))

    warns = con.execute(
        "SELECT kind, name, count, first_file, first_line FROM schema_warnings "
        "ORDER BY kind, count DESC").fetchall()
    if warns:
        print("\nSchema warnings (unknown fields — format may have changed):")
        for w in warns:
            print(f"  {w['kind']:>14}  {w['name']:<30} x{w['count']}  "
                  f"first: {w['first_file']}:{w['first_line']}")
    con.close()


if __name__ == "__main__":
    main()
