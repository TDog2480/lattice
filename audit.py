"""Phase 5: audit mode — raw evidence for any single detector finding.

Usage:
    python audit.py                 # finding counts per detector
    python audit.py d1              # indexed one-line list of D1 findings
    python audit.py d1 0            # full evidence for D1 finding #0
    python audit.py d4 3 --db lattice.db

For each finding this prints: session id, line numbers (the turn indices in the
source JSONL), timestamps, the sequence of surrounding events/tool calls, and a
re-run of the detector's filter logic showing why each exclusion did or did not
fire. Content is never printed — paths, digests, counts, and token numbers only.

To hand-verify against the source: the line numbers printed here are 1-based
line numbers in  ~/.claude/projects/<project>/<session_id>.jsonl
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from detectors import (D1_DROP_TOLERANCE, D1_TTL_SECONDS, D5_SMALL_EDIT_LINES,
                       parse_ts, run_all)

DETECTORS = {"d1": 0, "d2": 1, "d3": 2, "d4": 3, "d5": 4, "d6": 5, "d7": 6}


def open_db(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def session_header(con, sid):
    s = con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not s:
        print(f"session {sid}: NOT IN STORE")
        return
    print(f"session   {s['id']}")
    print(f"project   {s['project']}   branch={s['git_branch']}")
    print(f"span      {s['first_ts']} .. {s['last_ts']}")
    print(f"models    {s['models']}   api_calls={s['n_api_calls']} "
          f"tool_calls={s['n_tool_calls']}")
    print(f"source    ~/.claude/projects/{s['project']}/{s['id']}.jsonl")


def timeline(con, sid, center, before=10, after=10):
    """Merged event sequence around a line number. One row per source line."""
    lo, hi = 0, 10 ** 9
    rows = []
    for r in con.execute(
            """SELECT line_no, ts, model, input_tokens i, output_tokens o,
                      cache_creation cc, cache_read cr, cache_miss_type
               FROM usage WHERE session_id=?""", (sid,)):
        rows.append((r["line_no"], r["ts"] or "",
                     f"api_call     model={r['model']} in={r['i']} out={r['o']} "
                     f"cache_create={r['cc']:,} cache_read={r['cr']:,}"
                     + (f"  MISS={r['cache_miss_type']}" if r["cache_miss_type"] else "")))
    for r in con.execute(
            """SELECT line_no, ts, tool_name, target_path, command_head
               FROM tool_calls WHERE session_id=?""", (sid,)):
        tgt = r["target_path"] or r["command_head"] or ""
        rows.append((r["line_no"], r["ts"] or "",
                     f"tool_use     {r['tool_name']} {tgt}"))
    for r in con.execute(
            """SELECT line_no, ts, est_tokens, is_error FROM tool_results
               WHERE session_id=?""", (sid,)):
        rows.append((r["line_no"], r["ts"] or "",
                     f"tool_result  est_tokens={r['est_tokens']}"
                     + ("  ERROR" if r["is_error"] else "")))
    for r in con.execute(
            """SELECT line_no, ts, path, operation, read_offset, read_limit,
                      num_lines, total_lines, patch_lines, content_digest
               FROM file_events WHERE session_id=?""", (sid,)):
        if r["operation"] == "read":
            win = f"offset={r['read_offset']} limit={r['read_limit']}" \
                if (r["read_offset"] or r["read_limit"]) else "whole-file"
            det = f"{win} lines={r['num_lines']}/{r['total_lines']} " \
                  f"digest={r['content_digest']}"
        else:
            det = f"lines_touched={r['patch_lines']}"
        rows.append((r["line_no"], r["ts"] or "",
                     f"file_{r['operation']:<7} {r['path']}  {det}"))
    for r in con.execute(
            """SELECT line_no, ts, kind FROM events WHERE session_id=?""", (sid,)):
        rows.append((r["line_no"], r["ts"] or "", f"event        {r['kind']}"))
    for r in con.execute(
            """SELECT line_no, ts, is_exploratory FROM turns
               WHERE session_id=? AND is_human=1""", (sid,)):
        rows.append((r["line_no"], r["ts"] or "",
                     "HUMAN PROMPT"
                     + ("  (exploratory)" if r["is_exploratory"] else "")))

    rows.sort()
    idx = [i for i, r in enumerate(rows) if r[0] >= center]
    c = idx[0] if idx else len(rows) - 1
    window = rows[max(0, c - before): c + after + 1]
    print(f"\n-- timeline around line {center} "
          f"(showing {len(window)} of {len(rows)} events) --")
    for ln, ts, desc in window:
        mark = ">>" if ln == center else "  "
        print(f" {mark} L{ln:<5} {ts[11:19] if len(ts) > 19 else ts:<9} {desc}")


# -- per-detector audits ------------------------------------------------------

def audit_d1(con, f):
    sid = f["session"]
    session_header(con, sid)
    print(f"\nFINDING: cache invalidation at line {f['line_no']} "
          f"({f['rewritten_tokens']:,} tokens rewritten, ${f['usd']})")
    rows = con.execute(
        """SELECT line_no, ts, model, cache_creation cc, cache_read cr,
                  input_tokens i, cache_miss_type, cache_miss_tokens
           FROM usage WHERE session_id=? ORDER BY line_no""", (sid,)).fetchall()
    prev = None
    for r in rows:
        if r["line_no"] == f["line_no"]:
            break
        prev = r
    cur = next(r for r in rows if r["line_no"] == f["line_no"])
    print("\n-- the two API calls --")
    for label, r in (("prev", prev), ("THIS", cur)):
        print(f"  {label}: line={r['line_no']} ts={r['ts']} model={r['model']} "
              f"cache_read={r['cr']:,} cache_create={r['cc']:,} input={r['i']}")
    print("\n-- detector arithmetic --")
    p_ctx = (prev["cr"] or 0) + (prev["cc"] or 0)
    drop = (prev["cr"] or 0) - (cur["cr"] or 0)
    print(f"  prev cached context = prev.cache_read + prev.cache_create "
          f"= {prev['cr']:,} + {prev['cc']:,} = {p_ctx:,}")
    print(f"  cache_read drop     = {prev['cr']:,} - {cur['cr']:,} = {drop:,} "
          f"(> tolerance {D1_DROP_TOLERANCE} => candidate)")
    t1, t2 = parse_ts(prev["ts"] or ""), parse_ts(cur["ts"] or "")
    gap = (t2 - t1) if t1 and t2 else None
    print(f"  gap between calls   = {gap:.0f}s "
          f"({'<' if gap and gap <= D1_TTL_SECONDS else '>'} TTL {D1_TTL_SECONDS}s "
          f"=> {'counted' if gap and gap <= D1_TTL_SECONDS else 'would be excluded as TTL expiry'})")
    print(f"  rewritten           = min(cur.cache_create, prev_ctx - cur.cache_read) "
          f"= min({cur['cc']:,}, {p_ctx - (cur['cr'] or 0):,}) "
          f"= {f['rewritten_tokens']:,}")
    print(f"  API-labeled cause   = {cur['cache_miss_type']} "
          f"(cache_missed_input_tokens={cur['cache_miss_tokens']})")
    print("  pricing             = rewritten * (1h-write - cache-read) rate "
          "for the model on THIS call")
    timeline(con, sid, f["line_no"], before=6, after=6)


def audit_d3(con, f):
    session_header(con, f["session"])
    print(f"\nFINDING: tool result of ~{f['est_tokens']:,} est. tokens "
          f"(excess over 10k: {f['excess_tokens']:,}) at line {f['line_no']}")
    print(f"tool={f['tool']} command_head={f['command_head']} "
          f"target={f['target_path']}")
    print("basis: chars/4 ESTIMATE — transcript records no token count")
    timeline(con, f["session"], f["line_no"])


def audit_d4(con, f):
    if f.get("scope") == "cross_session":
        print(f"CROSS-SESSION finding: {f['path']}")
        print(f"read in {f['sessions_reading']} sessions of project "
              f"{f['project']} (~{f['est_tokens_repeat_reads']:,} est. tokens "
              "in repeat sessions)")
        print("classified as OPPORTUNITY, excluded from waste totals: a fresh "
              "session has an empty context, so each first-read is locally "
              "necessary.\n")
        for sid in f["sessions"]:
            r = con.execute(
                """SELECT line_no, ts, read_offset, read_limit, content_chars,
                          content_digest FROM file_events
                   WHERE session_id=? AND path=? AND operation='read'
                   ORDER BY line_no LIMIT 1""", (sid, f["path"])).fetchone()
            print(f"  {sid}  first read L{r['line_no']} {r['ts']} "
                  f"offset={r['read_offset']} limit={r['read_limit']} "
                  f"chars={r['content_chars']} digest={r['content_digest']}")
        print("\nnote: digests differing across sessions => the file changed "
              "between sessions; memory would need invalidation, and the "
              "opportunity estimate above is an upper bound.")
        return

    sid = f["session"]
    session_header(con, sid)
    print(f"\nFINDING: within-session repeat read of {f['path']} "
          f"(window offset={f['offset']} limit={f['limit']}) "
          f"first L{f['first_read_line']} repeat L{f['repeat_line']} "
          f"~{f['est_tokens']:,} est. tokens")
    _d4_filter_trace(con, sid, f["path"], f["offset"], f["limit"])
    timeline(con, sid, f["repeat_line"], before=8, after=4)


def _d4_filter_trace(con, sid, path, off, lim):
    """Re-run the D4 exclusion cascade over every read pair of this window."""
    reads = con.execute(
        """SELECT fe.line_no, fe.ts, fe.content_digest,
                  COALESCE(t.is_sidechain,0) side
           FROM file_events fe
           LEFT JOIN tool_calls tc ON tc.id = fe.tool_use_id
           LEFT JOIN turns t ON t.uuid = tc.entry_uuid
           WHERE fe.session_id=? AND fe.path=? AND fe.operation='read'
             AND fe.read_offset IS ? AND fe.read_limit IS ?
           ORDER BY fe.line_no""", (sid, path, off, lim)).fetchall()
    edits = [r["line_no"] for r in con.execute(
        """SELECT line_no FROM file_events WHERE session_id=? AND path=?
           AND operation IN ('edit','write') ORDER BY line_no""", (sid, path))]
    print(f"\n-- filter trace: {len(reads)} reads of this window, "
          f"edits to path at lines {edits or 'none'} --")
    for prev, cur in zip(reads, reads[1:]):
        between = [ln for ln in edits if prev["line_no"] < ln < cur["line_no"]]
        if cur["side"]:
            verdict = "EXCLUDED (c): read inside subagent context"
        elif between:
            verdict = f"EXCLUDED (a): edit(s) at line {between} between reads " \
                      "=> verification read"
        elif prev["content_digest"] != cur["content_digest"]:
            verdict = "EXCLUDED (d): content digest changed => file changed on disk"
        else:
            verdict = "COUNTED AS WASTE: identical content, no edit between, " \
                      "not a subagent. (b) compaction not checkable — no " \
                      "compaction events exist in this data."
        print(f"  L{prev['line_no']} -> L{cur['line_no']}: "
              f"digests {prev['content_digest']} -> {cur['content_digest']}: "
              f"{verdict}")


def audit_d5(con, f):
    sid = f["session"]
    session_header(con, sid)
    print(f"\nFINDING: whole-file read of {f['path']} at L{f['read_line']} "
          f"(~{f['read_est_tokens']:,} est. tokens, {f['file_lines']} lines) "
          f"followed by edit at L{f['edit_line_no']} touching only "
          f"{f['edit_lines_touched']} lines (threshold {D5_SMALL_EDIT_LINES})")
    print("labeled UPPER BOUND: correctness of a small edit may legitimately "
          "require the whole file; the needed fraction is unknowable here.")
    timeline(con, sid, f["read_line"], before=4, after=10)


def audit_d6(con, f):
    sid = f["session"]
    session_header(con, sid)
    print(f"\nFINDING: {f['consecutive_searches']} consecutive Grep/Glob "
          f"before first Read in the task starting at prompt L{f['task_start_line']}")
    print(f"search calls at lines {f['search_lines']}; tokens counted (runs "
          f"beyond the 2nd search): {f['est_tokens_beyond_two']:,}")
    p = con.execute(
        "SELECT is_exploratory FROM turns WHERE session_id=? AND line_no=?",
        (sid, f["task_start_line"])).fetchone()
    print(f"exploratory-prompt exclusion: is_exploratory="
          f"{p['is_exploratory'] if p else '?'} => "
          f"{'would have been excluded' if p and p['is_exploratory'] else 'not excluded'}")
    timeline(con, sid, f["search_lines"][0], before=3, after=12)


def audit_d2(con, f):
    print(f"project {f['project']}: {f['available']} tools declared available, "
          f"{f['invoked']} invoked, {f['never_invoked']} never invoked "
          f"({f['mcp_never_invoked']} of them MCP).")
    print("evidence: declaration lines (attachment:deferred_tools_delta) per session:\n")
    for r in con.execute(
            """SELECT ta.session_id, MIN(ta.line_no) line_no, COUNT(*) n
               FROM tool_availability ta JOIN sessions s ON s.id=ta.session_id
               WHERE s.project=? AND ta.action='added'
               GROUP BY ta.session_id""", (f["project"],)):
        print(f"  {r['session_id']}  first delta at L{r['line_no']} "
              f"({r['n']} names added)")
    print(f"\nnever invoked: {', '.join(f['never_invoked_names'])}")
    print("\nreminder: token cost NOT MEASURABLE — names only, schemas are "
          "deferred and never appear in transcripts.")


AUDITORS = {"d1": audit_d1, "d2": audit_d2, "d3": audit_d3,
            "d4": audit_d4, "d5": audit_d5, "d6": audit_d6}


def one_liner(det_key, i, f):
    if det_key == "d1":
        return (f"[{i}] {f['session'][:8]} L{f['line_no']} "
                f"{f['rewritten_tokens']:>9,} tok  ${f['usd']:<8} "
                f"cause={f['labeled_reason']}")
    if det_key == "d4":
        if f.get("scope") == "cross_session":
            return (f"[{i}] cross-session {f['path']}  "
                    f"{f['sessions_reading']} sessions")
        return (f"[{i}] within {f['session'][:8]} {f['path']} "
                f"L{f['first_read_line']}->L{f['repeat_line']}")
    if det_key == "d2":
        return f"[{i}] {f['project']}  unused={f['never_invoked']}"
    if det_key == "d6":
        return (f"[{i}] {f['session'][:8]} prompt L{f['task_start_line']} "
                f"{f['consecutive_searches']} searches")
    if det_key == "d3":
        return (f"[{i}] {f['session'][:8]} L{f['line_no']} "
                f"{f['est_tokens']:,} tok {f['tool']}")
    if det_key == "d5":
        return (f"[{i}] {f['session'][:8]} {f['path']} "
                f"read L{f['read_line']} ~{f['read_est_tokens']:,} tok")
    return f"[{i}] {f}"


def main():
    ap = argparse.ArgumentParser(usage="audit.py [d1..d7] [index] [--db DB]")
    ap.add_argument("detector", nargs="?", help="d1..d7")
    ap.add_argument("index", nargs="?", type=int, help="finding index")
    ap.add_argument("--db", default="lattice.db")
    args = ap.parse_args()

    results = run_all(args.db)
    con = open_db(args.db)

    if args.detector is None:
        for key, pos in DETECTORS.items():
            r = results[pos]
            print(f"{key}: {r['category']:<40} findings={len(r['findings'])} "
                  f"measurable={r['measurable']}")
        print("\nusage: python audit.py d1        (list)")
        print("       python audit.py d1 0      (evidence for finding #0)")
        return

    key = args.detector.lower()
    if key not in DETECTORS:
        sys.exit(f"unknown detector {key!r}; use d1..d7")
    r = results[DETECTORS[key]]
    findings = r["findings"]

    if args.index is None:
        print(f"{r['category']} — {len(findings)} findings")
        for c in r["caveats"]:
            print(f"  caveat: {c}")
        print()
        for i, f in enumerate(findings):
            print(one_liner(key, i, f))
        if not findings:
            print("(no findings — nothing to audit; that is the result)")
        return

    if not 0 <= args.index < len(findings):
        sys.exit(f"index out of range: {key} has {len(findings)} findings")
    print("=" * 78)
    AUDITORS[key](con, findings[args.index])
    con.close()


if __name__ == "__main__":
    main()
