"""Phase 1: schema discovery for Claude Code session transcripts.

Scans ~/.claude/projects/**/*.jsonl read-only and prints a report answering:
  1. file inventory            2. top-level keys
  3. entry types + examples    4. where token usage lives (per-call vs cumulative)
  5. tool definitions present? 6. compaction representation
  7. subagent distinction      8. resume/fork linkage
  9. things asked about that were not found

Never modifies source files. Redacts all content strings before printing.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Keys whose string values are structural metadata, safe to print (truncated).
SAFE_KEYS = {
    "type", "subtype", "role", "model", "operation", "name", "stop_reason",
    "stop_sequence", "version", "userType", "entrypoint", "permissionMode",
    "kind", "promptSource", "level", "status", "sessionId", "uuid",
    "parentUuid", "promptId", "requestId", "timestamp", "cwd", "gitBranch",
    "id", "tool_use_id", "toolUseID", "slug", "service_tier", "leafUuid",
    "logicalParentUuid", "sourceSessionId", "compactMetadata", "hook_event_name",
    "media_type", "source", "mode", "agentType", "agentId", "label",
}

TOKEN_FIELD_RE = re.compile(r"token|cost|usd", re.IGNORECASE)
USAGE_FIELDS = (
    "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
)


def redact(obj, depth=0):
    """Structure-preserving redaction: keep keys, types, sizes; drop content."""
    if depth > 7:
        return "<max-depth>"
    if isinstance(obj, dict):
        return {k: redact(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [redact(v, depth + 1) for v in obj[:2]]
        if len(obj) > 2:
            out.append(f"<...+{len(obj) - 2} more>")
        return out
    if isinstance(obj, str):
        return obj[:60] if len(obj) <= 60 else f"<str:{len(obj)}>"
    return obj  # numbers, bools, null


def redact_entry(entry):
    """Redact but keep SAFE_KEYS values readable at any depth."""
    def walk(obj, key=None, depth=0):
        if depth > 7:
            return "<max-depth>"
        if isinstance(obj, dict):
            return {k: walk(v, k, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            out = [walk(v, key, depth + 1) for v in obj[:2]]
            if len(obj) > 2:
                out.append(f"<...+{len(obj) - 2} more>")
            return out
        if isinstance(obj, str):
            if key in SAFE_KEYS or len(obj) <= 24:
                return obj[:80]
            return f"<str:{len(obj)}>"
        return obj
    return walk(entry)


def json_paths(obj, prefix=""):
    """Yield (generalized_path, leaf_value) pairs; array indices become []."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from json_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_paths(v, f"{prefix}[]")
    else:
        yield prefix, obj


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    if not PROJECTS_DIR.is_dir():
        sys.exit(f"not found: {PROJECTS_DIR}")

    files = sorted(PROJECTS_DIR.rglob("*.jsonl"))

    # -- accumulators -------------------------------------------------------
    inventory = defaultdict(lambda: {"files": 0, "bytes": 0, "lines": 0,
                                     "malformed": 0, "ts_min": None, "ts_max": None})
    top_keys = Counter()
    entry_types = Counter()
    entry_examples = {}
    usage_paths = Counter()            # generalized path -> count
    usage_by_entry_type = Counter()    # entry type carrying usage
    usage_sequences = defaultdict(list)  # file -> [(input, out, cc, cr, requestId)]
    tool_def_evidence = []             # places where input_schema / description schemas appear
    deferred_tool_names = defaultdict(set)  # project -> tool names from deferred_tools_delta
    tool_use_names = Counter()
    compact_hits = []                  # (file, line_no, redacted entry) for compaction-ish entries
    compact_keys = Counter()
    sidechain_counts = Counter()       # file -> sidechain line count
    sidechain_types = Counter()
    session_ids_per_file = defaultdict(set)
    file_per_session_id = defaultdict(set)
    link_keys = Counter()              # keys suggesting resume/fork linkage
    link_examples = {}
    versions = Counter()
    models = Counter()
    sidecar_dirs = []

    for d in sorted(PROJECTS_DIR.iterdir()):
        if d.is_dir():
            for sub in d.iterdir():
                if sub.is_dir() and sub.name != "memory":
                    sidecar_dirs.append(sub)

    for f in files:
        project = f.parent.name if f.parent != PROJECTS_DIR else "<root>"
        inv = inventory[project]
        inv["files"] += 1
        inv["bytes"] += f.stat().st_size
        stem = f.stem

        for line_no, raw in enumerate(f.open(encoding="utf-8", errors="replace"), 1):
            raw = raw.strip()
            if not raw:
                continue
            inv["lines"] += 1
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                inv["malformed"] += 1
                continue
            if not isinstance(e, dict):
                inv["malformed"] += 1
                continue

            top_keys.update(e.keys())
            etype = e.get("type", "<missing>")
            # attachments have their own inner type worth distinguishing
            if etype == "user" and isinstance(e.get("attachment"), dict):
                etype = f"attachment:{e['attachment'].get('type', '?')}"
            entry_types[etype] += 1
            if etype not in entry_examples:
                entry_examples[etype] = (f.name, line_no, redact_entry(e))

            ts = parse_ts(e.get("timestamp", "")) if isinstance(e.get("timestamp"), str) else None
            if ts:
                if inv["ts_min"] is None or ts < inv["ts_min"]:
                    inv["ts_min"] = ts
                if inv["ts_max"] is None or ts > inv["ts_max"]:
                    inv["ts_max"] = ts

            if isinstance(e.get("version"), str):
                versions[e["version"]] += 1
            sid = e.get("sessionId")
            if isinstance(sid, str):
                session_ids_per_file[f].add(sid)
                file_per_session_id[sid].add(f.name)

            # linkage-suggesting keys
            for k in e:
                if re.search(r"resume|fork|branch|logicalparent|source.*session|compact",
                             k, re.IGNORECASE):
                    link_keys[k] += 1
                    link_examples.setdefault(k, (f.name, line_no, redact_entry(e)))

            # sidechain / subagent markers
            if e.get("isSidechain"):
                sidechain_counts[f.name] += 1
                sidechain_types[e.get("type", "?")] += 1

            # scan all paths once per entry
            found_usage_here = False
            for path, val in json_paths(e):
                leaf = path.rsplit(".", 1)[-1]
                if TOKEN_FIELD_RE.search(leaf) or ".usage." in path:
                    usage_paths[path] += 1
                    found_usage_here = True
                if leaf == "input_schema" or path.endswith("tools[].input_schema"):
                    tool_def_evidence.append((f.name, line_no, path))
                if path.endswith("compactMetadata") or "compact" in path.lower() \
                        or (leaf == "subtype" and isinstance(val, str) and "compact" in val):
                    compact_keys[path] += 1
            if found_usage_here:
                usage_by_entry_type[e.get("type", "?")] += 1

            # compaction: entry types / subtypes / flags
            blob_keys = " ".join(e.keys())
            if e.get("type") == "summary" or e.get("isCompactSummary") \
                    or "compact" in blob_keys.lower() \
                    or (isinstance(e.get("subtype"), str) and "compact" in e["subtype"]):
                if len(compact_hits) < 6:
                    compact_hits.append((f.name, line_no, redact_entry(e)))

            # message-level extraction
            msg = e.get("message")
            if isinstance(msg, dict):
                if isinstance(msg.get("model"), str):
                    models[msg["model"]] += 1
                u = msg.get("usage")
                if isinstance(u, dict) and any(k in u for k in USAGE_FIELDS):
                    usage_sequences[f.name].append((
                        line_no,
                        u.get("input_tokens"), u.get("output_tokens"),
                        u.get("cache_creation_input_tokens"),
                        u.get("cache_read_input_tokens"),
                        e.get("requestId"),
                    ))
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                tool_use_names[block.get("name", "?")] += 1
                # tool definitions would live in a top-level "tools" param
                for tk in ("tools", "tool_choice", "system"):
                    if tk in msg:
                        tool_def_evidence.append((f.name, line_no, f"message.{tk}"))

            # deferred tools attachment: the only tool-availability signal
            att = e.get("attachment")
            if isinstance(att, dict) and att.get("type") == "deferred_tools_delta":
                for n in att.get("addedNames", []):
                    deferred_tool_names[project].add(n)

    # ======================================================================
    P = print
    P("# Claude Code transcript schema discovery")
    P(f"\nScanned: {PROJECTS_DIR}  ({len(files)} jsonl files)\n")

    # -- 1. inventory -------------------------------------------------------
    P("## 1. File inventory\n")
    P("| project | files | size KB | lines | malformed | first ts | last ts |")
    P("|---|---|---|---|---|---|---|")
    tot_b = tot_l = tot_m = 0
    for proj, inv in sorted(inventory.items()):
        tot_b += inv["bytes"]; tot_l += inv["lines"]; tot_m += inv["malformed"]
        fmt = lambda t: t.strftime("%Y-%m-%d") if t else "-"
        P(f"| {proj} | {inv['files']} | {inv['bytes']//1024} | {inv['lines']} "
          f"| {inv['malformed']} | {fmt(inv['ts_min'])} | {fmt(inv['ts_max'])} |")
    P(f"| **total** | {len(files)} | {tot_b//1024} | {tot_l} | {tot_m} | | |")
    if sidecar_dirs:
        P("\nSidecar dirs (non-JSONL payloads, e.g. fetched files):")
        for d in sidecar_dirs:
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            P(f"  - {d.relative_to(PROJECTS_DIR)}  ({n} files)")

    # -- 2. top-level keys --------------------------------------------------
    P("\n## 2. Top-level keys (frequency)\n")
    for k, c in top_keys.most_common():
        P(f"  {c:>7}  {k}")

    # -- 3. entry types -----------------------------------------------------
    P("\n## 3. Entry types\n")
    for t, c in entry_types.most_common():
        P(f"  {c:>7}  {t}")
    P("\n### One redacted example per type\n")
    for t, (fn, ln, ex) in sorted(entry_examples.items()):
        P(f"--- type={t}  ({fn}:{ln})")
        s = json.dumps(ex, indent=1, default=str)
        P(s[:1400] + ("\n  ...<truncated>" if len(s) > 1400 else ""))

    # -- 4. token usage location -------------------------------------------
    P("\n## 4. Token usage fields\n")
    P("Distinct JSON paths containing token/cost fields (path -> occurrences):")
    for p, c in sorted(usage_paths.items()):
        P(f"  {c:>7}  {p}")
    P("\nEntry types carrying usage:", dict(usage_by_entry_type))
    P("\nPer-call vs cumulative check (input should track context size per call;")
    P("output should be small and non-monotonic if per-call):")
    inc_in = dec_in = inc_out = dec_out = 0
    for fn, seq in usage_sequences.items():
        for (a, b) in zip(seq, seq[1:]):
            if a[1] is not None and b[1] is not None:
                inc_in += b[1] >= a[1]; dec_in += b[1] < a[1]
            if a[2] is not None and b[2] is not None:
                inc_out += b[2] >= a[2]; dec_out += b[2] < a[2]
    P(f"  input_tokens:  {inc_in} non-decreasing steps, {dec_in} decreasing")
    P(f"  output_tokens: {inc_out} non-decreasing steps, {dec_out} decreasing")
    big = max(usage_sequences, key=lambda k: len(usage_sequences[k]), default=None)
    if big:
        P(f"\nSample sequence from {big} (line, in, out, cache_create, cache_read, requestId):")
        for row in usage_sequences[big][:12]:
            P(f"  {row}")
        dup_req = Counter(r[5] for r in usage_sequences[big] if r[5])
        dups = {k: v for k, v in dup_req.items() if v > 1}
        P(f"  entries sharing a requestId (same API call logged multiple times): "
          f"{len(dups)} requestIds, {sum(dups.values())} entries")

    # -- 5. tool definitions ------------------------------------------------
    P("\n## 5. Tool definitions vs invocations\n")
    if tool_def_evidence:
        P("Tool schema-like structures FOUND at:")
        for fn, ln, p in tool_def_evidence[:10]:
            P(f"  {fn}:{ln}  {p}")
    else:
        P("NO tool definitions (input_schema / tools array / system prompt) found")
        P("anywhere in the transcripts. Only invocations are recorded.")
    P(f"\nDistinct tools actually invoked ({len(tool_use_names)}):")
    for n, c in tool_use_names.most_common():
        P(f"  {c:>6}  {n}")
    if deferred_tool_names:
        P("\nTool NAMES known to be available via attachment:deferred_tools_delta")
        P("(names only — no schemas, so schema token cost is not in the data):")
        for proj, names in deferred_tool_names.items():
            P(f"  {proj}: {len(names)} names, e.g. {sorted(names)[:5]}")

    # -- 6. compaction ------------------------------------------------------
    P("\n## 6. Compaction\n")
    if compact_keys:
        P("Compaction-related paths:")
        for p, c in compact_keys.most_common():
            P(f"  {c:>6}  {p}")
    if compact_hits:
        P("\nExamples:")
        for fn, ln, ex in compact_hits:
            s = json.dumps(ex, default=str)
            P(f"  {fn}:{ln}  {s[:500]}")
    if not compact_keys and not compact_hits:
        P("NOT FOUND: no entry type, subtype, key, or flag mentioning compaction.")

    # -- 7. subagents -------------------------------------------------------
    P("\n## 7. Subagent distinction\n")
    n_side = sum(sidechain_counts.values())
    P(f"isSidechain=true entries: {n_side} across {len(sidechain_counts)} files")
    if sidechain_types:
        P(f"  by entry type: {dict(sidechain_types)}")
    agent_calls = {n: c for n, c in tool_use_names.items() if n in ("Task", "Agent")}
    P(f"Agent-spawn tool_use calls (Task/Agent): {agent_calls or 'none'}")
    multi = {str(f.name): sorted(s) for f, s in session_ids_per_file.items() if len(s) > 1}
    P(f"Files containing >1 sessionId: {len(multi)}")
    for fn, sids in list(multi.items())[:5]:
        P(f"  {fn}: {sids}")

    # -- 8. resume / fork linkage ------------------------------------------
    P("\n## 8. Resume / fork linkage\n")
    mismatched = {str(f.name): sorted(s) for f, s in session_ids_per_file.items()
                  if f.stem not in s}
    P(f"Files whose name != any sessionId inside: {len(mismatched)}")
    shared = {sid: sorted(fs) for sid, fs in file_per_session_id.items() if len(fs) > 1}
    P(f"sessionIds appearing in >1 file: {len(shared)}")
    for sid, fs in list(shared.items())[:5]:
        P(f"  {sid}: {fs}")
    if link_keys:
        P("Keys suggesting linkage/resume/fork:")
        for k, c in link_keys.most_common():
            P(f"  {c:>6}  {k}")
        for k, (fn, ln, ex) in list(link_examples.items())[:4]:
            P(f"  example {k} ({fn}:{ln}): {json.dumps(ex, default=str)[:400]}")
    else:
        P("No resume/fork/branch keys observed.")

    # -- misc + 9 -----------------------------------------------------------
    P("\n## Versions and models observed\n")
    for v, c in versions.most_common():
        P(f"  {c:>7}  version {v}")
    for m, c in models.most_common():
        P(f"  {c:>7}  model {m}")

    P("\n## 9. Asked about but NOT found (computed)\n")
    if not tool_def_evidence:
        P("- Tool definitions/schemas: absent. MCP schema overhead is NOT directly")
        P("  measurable; only tool NAMES (deferred_tools_delta) and invocations exist.")
    if not compact_keys and not compact_hits:
        P("- Compaction events: no representation found.")
    P("- Anything else absent is visible by omission in sections 2-8 above.")


if __name__ == "__main__":
    main()
