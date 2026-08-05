"""Phase 3: waste detectors over the lattice.db store.

Each detector is a pure function over an open sqlite connection returning:

    {
      "category":    str,
      "measurable":  bool,          # False => no token/dollar estimate possible
      "waste_tokens": int,          # estimated wasted tokens (0 if not measurable)
      "waste_usd":   float,
      "confidence":  "high" | "medium" | "low" | "n/a",
      "estimated":   bool,          # True when tokens are chars/4 estimates
      "caveats":     [str],         # reasons the number may be inflated/deflated
      "stats":       {...},         # distributions and context
      "findings":    [ {...} ],     # EVIDENCE: session ids, line numbers, paths
    }

Run: python detectors.py [--db lattice.db] [--json findings.json]

Token pricing is configurable below. Dollar figures are NOTIONAL — this data
comes from a Claude Code subscription, not metered API billing; dollars answer
"what would this cost at API rates", nothing more.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict

# -- pricing (USD per million tokens), current API rates as of 2026-08-05 -----
# Sonnet 5 is at introductory pricing ($2/$10) through 2026-08-31, which covers
# this dataset's entire date range; the standard rate is $3/$15.
# Cache multipliers: write 5m = 1.25x input, write 1h = 2x input, read = 0.1x.
PRICING = {
    "claude-sonnet-5":   {"input": 2.00,  "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-fable-5":    {"input": 10.00, "output": 50.00},
    "_default":          {"input": 2.00,  "output": 10.00},
}
CACHE_W_5M, CACHE_W_1H, CACHE_READ = 1.25, 2.0, 0.10

# D1: cache_read drops below the previous call by more than this => prefix
# invalidation (small jitter tolerated)
D1_DROP_TOLERANCE = 100
D1_TTL_SECONDS = 3600          # gaps longer than the 1h cache TTL are expiry, not a bug
D3_THRESHOLD_TOKENS = 10_000
D5_MIN_READ_CHARS = 12_000     # ~3k tokens
D5_SMALL_EDIT_LINES = 40
D6_RUN_LENGTH = 3


def rate(model: str | None) -> dict:
    return PRICING.get(model or "", PRICING["_default"])


def usd(tokens: float, model: str | None, kind: str) -> float:
    """kind: input | output | cache_read | cache_w_1h | cache_w_5m"""
    r = rate(model)
    mult = {"input": 1.0, "output": None, "cache_read": CACHE_READ,
            "cache_w_1h": CACHE_W_1H, "cache_w_5m": CACHE_W_5M}[kind]
    base = r["output"] if kind == "output" else r["input"] * (mult or 1.0)
    return tokens * base / 1_000_000


def session_model(con, sid: str) -> str | None:
    r = con.execute(
        """SELECT model, COUNT(*) n FROM usage WHERE session_id=? AND model != '<synthetic>'
           GROUP BY model ORDER BY n DESC LIMIT 1""", (sid,)).fetchone()
    return r["model"] if r else None


def quantiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "min": round(s[0], 4), "p25": round(q(0.25), 4),
            "median": round(q(0.5), 4), "p75": round(q(0.75), 4),
            "max": round(s[-1], 4)}


def parse_ts(ts):
    from datetime import datetime
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


# ============================================================================
# D1 — cache efficiency
# ============================================================================

def d1_cache_efficiency(con) -> dict:
    findings, hit_rates = [], []
    total_waste_tok, total_waste_usd = 0, 0.0
    n_invalidations = n_ttl_expiry = 0
    miss_reasons = defaultdict(int)
    uncached_input_total = 0

    sessions = con.execute(
        "SELECT id FROM sessions WHERE n_api_calls >= 3").fetchall()
    for s in sessions:
        sid = s["id"]
        rows = con.execute(
            """SELECT line_no, ts, model, input_tokens i, output_tokens o,
                      cache_creation cc, cache_read cr, eph_1h, eph_5m,
                      cache_miss_type, cache_miss_tokens
               FROM usage WHERE session_id=? AND model != '<synthetic>'
               ORDER BY line_no""", (sid,)).fetchall()
        if len(rows) < 3:
            continue
        body = rows[1:]  # exclude call 1: always creates cache
        denom = sum((r["cr"] or 0) + (r["cc"] or 0) + (r["i"] or 0) for r in body)
        if denom:
            hit_rates.append(sum(r["cr"] or 0 for r in body) / denom)
        uncached_input_total += sum(r["i"] or 0 for r in body)

        for prev, cur in zip(rows, rows[1:]):
            if cur["cache_miss_type"]:
                miss_reasons[cur["cache_miss_type"]] += 1
            p_cr, c_cr = prev["cr"] or 0, cur["cr"] or 0
            if c_cr >= p_cr - D1_DROP_TOLERANCE:
                continue
            t_prev, t_cur = parse_ts(prev["ts"] or ""), parse_ts(cur["ts"] or "")
            if t_prev and t_cur and (t_cur - t_prev) > D1_TTL_SECONDS:
                n_ttl_expiry += 1
                continue
            # prefix invalidated: tokens re-written that were previously cached
            prev_ctx = p_cr + (prev["cc"] or 0)
            rewritten = min(cur["cc"] or 0, max(0, prev_ctx - c_cr))
            if rewritten <= 0:
                continue
            n_invalidations += 1
            # cost above the cache-read price those tokens would otherwise pay
            w = usd(rewritten, cur["model"], "cache_w_1h") - \
                usd(rewritten, cur["model"], "cache_read")
            total_waste_tok += rewritten
            total_waste_usd += w
            findings.append({
                "session": sid, "line_no": cur["line_no"], "ts": cur["ts"],
                "cache_read_drop": p_cr - c_cr, "rewritten_tokens": rewritten,
                "usd": round(w, 4),
                "labeled_reason": cur["cache_miss_type"],
            })

    findings.sort(key=lambda f: -f["rewritten_tokens"])
    return {
        "category": "D1 cache efficiency", "measurable": True,
        "waste_tokens": total_waste_tok, "waste_usd": round(total_waste_usd, 2),
        "confidence": "high", "estimated": False,
        "caveats": [
            "Waste = tokens re-written to cache after a mid-session prefix "
            "invalidation, priced at (1h-write - cache-read) rates. Assumes the "
            "re-written span would otherwise have stayed cached — true unless "
            "the session was about to end anyway.",
            "Drops after a >1h gap are counted as TTL expiry (unavoidable), "
            f"not waste: {0} of those included.",
            "input_tokens (fully uncached) are negligible in this dataset — "
            "caching is working; this measures the exceptions.",
        ],
        "stats": {
            "sessions_analyzed": len(hit_rates),
            "hit_rate_distribution": quantiles(hit_rates),
            "mid_session_invalidations": n_invalidations,
            "ttl_expiry_drops_excluded": n_ttl_expiry,
            "uncached_input_tokens_total": uncached_input_total,
            "api_labeled_miss_reasons": dict(miss_reasons),
        },
        "findings": findings,
    }


# ============================================================================
# D2 — unused tools
# ============================================================================

def d2_unused_tools(con) -> dict:
    per_project = {}
    for p in con.execute("SELECT DISTINCT project FROM sessions"):
        proj = p["project"]
        avail = {r["tool_name"] for r in con.execute(
            """SELECT DISTINCT ta.tool_name FROM tool_availability ta
               JOIN sessions s ON s.id = ta.session_id
               WHERE s.project=? AND ta.action='added'""", (proj,))}
        used = {r["tool_name"] for r in con.execute(
            """SELECT DISTINCT tc.tool_name FROM tool_calls tc
               JOIN sessions s ON s.id = tc.session_id WHERE s.project=?""",
            (proj,))}
        unused = sorted(avail - used)
        mcp_unused = [t for t in unused if t.startswith("mcp__")]
        per_project[proj] = {
            "available": len(avail), "invoked": len(used & avail),
            "never_invoked": len(unused),
            "never_invoked_names": unused,
            "mcp_never_invoked": len(mcp_unused),
        }
    return {
        "category": "D2 unused tools", "measurable": False,
        "waste_tokens": 0, "waste_usd": 0.0, "confidence": "n/a",
        "estimated": False,
        "caveats": [
            "TOKEN COST NOT MEASURABLE FROM THIS DATA: transcripts contain tool "
            "NAMES (deferred_tools_delta) but never tool schemas, so the context "
            "cost of unused definitions cannot be computed. Measuring it would "
            "require capturing the actual API request payload (e.g. a proxy) or "
            "vendor documentation of per-tool schema sizes.",
            "The names in deferred_tools_delta are DEFERRED tools: this Claude "
            "Code version loads only names into context and fetches schemas "
            "on demand via ToolSearch. The classic 'unused MCP schemas bloat "
            "every prompt' cost is therefore largely already mitigated here — "
            "counts below are exposure, not measured waste.",
        ],
        "stats": {"per_project": per_project},
        "findings": [
            {"project": proj, **info} for proj, info in per_project.items()
        ],
    }


# ============================================================================
# D3 — oversized tool output
# ============================================================================

def d3_oversized_output(con) -> dict:
    rows = con.execute(
        """SELECT tr.session_id sid, tr.line_no, tr.est_tokens, tr.content_chars,
                  tc.tool_name, tc.command_head, tc.target_path
           FROM tool_results tr LEFT JOIN tool_calls tc ON tc.id = tr.tool_use_id
           WHERE tr.est_tokens > ?
           ORDER BY tr.est_tokens DESC""", (D3_THRESHOLD_TOKENS,)).fetchall()

    findings, groups = [], defaultdict(lambda: {"n": 0, "tokens": 0, "sessions": set()})
    waste_tok, waste_usd = 0, 0.0
    for r in rows:
        excess = r["est_tokens"] - D3_THRESHOLD_TOKENS
        m = session_model(con, r["sid"])
        w = usd(excess, m, "cache_w_1h")  # enters context once as a cache write
        waste_tok += excess
        waste_usd += w
        key = (r["tool_name"] or "?", r["command_head"] or r["target_path"] or "")
        g = groups[key]
        g["n"] += 1
        g["tokens"] += r["est_tokens"]
        g["sessions"].add(r["sid"])
        findings.append({
            "session": r["sid"], "line_no": r["line_no"],
            "tool": r["tool_name"], "command_head": r["command_head"],
            "target_path": r["target_path"],
            "est_tokens": r["est_tokens"], "excess_tokens": excess,
            "usd_excess": round(w, 4),
        })

    repeat = [{"tool": k[0], "source": k[1], "count": v["n"],
               "total_est_tokens": v["tokens"], "sessions": len(v["sessions"])}
              for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["tokens"])
              if v["n"] > 1]
    return {
        "category": "D3 oversized tool output", "measurable": True,
        "waste_tokens": waste_tok, "waste_usd": round(waste_usd, 2),
        "confidence": "medium", "estimated": True,
        "caveats": [
            "Token sizes are chars/4 ESTIMATES — the transcript does not record "
            "tool-result token counts.",
            "Waste counts only the excess above the 10k threshold, priced once "
            "as a 1h cache write. Real cost is higher: the excess is re-read "
            "(at 0.1x) by every subsequent call in the session. Not multiplied "
            "in, to stay conservative.",
            "Some large outputs are legitimately needed in full (e.g. reading "
            "a large file the model must edit). Judged mechanical, not verified "
            "per-case.",
            "DOUBLE-COUNT RISK: an oversized Read result can also appear in D4 "
            "(repeated reads) and D5 (whole-file reads). Do not sum D3+D4+D5 "
            "without dedup.",
        ],
        "stats": {"results_over_threshold": len(rows),
                  "repeat_offenders": repeat},
        "findings": findings,
    }


# ============================================================================
# D4 — repeated file reads
# ============================================================================

def d4_repeated_reads(con) -> dict:
    reads = con.execute(
        """SELECT fe.session_id sid, s.project, fe.line_no, fe.path,
                  fe.read_offset, fe.read_limit, fe.content_chars,
                  fe.content_digest, fe.ts,
                  COALESCE(t.is_sidechain, 0) sidechain
           FROM file_events fe
           JOIN sessions s ON s.id = fe.session_id
           LEFT JOIN tool_calls tc ON tc.id = fe.tool_use_id
           LEFT JOIN turns t ON t.uuid = tc.entry_uuid
           WHERE fe.operation = 'read'
           ORDER BY fe.session_id, fe.line_no""").fetchall()
    edits = con.execute(
        """SELECT session_id sid, line_no, path FROM file_events
           WHERE operation IN ('edit', 'write')""").fetchall()
    edits_by_key = defaultdict(list)
    for e in edits:
        edits_by_key[(e["sid"], e["path"])].append(e["line_no"])

    excl = {"verification_after_edit": 0, "compaction_boundary": "NOT DETECTABLE",
            "subagent_context": 0, "file_changed_on_disk": 0,
            "different_window": 0}
    findings = []
    waste_tok, waste_usd = 0, 0.0

    within = defaultdict(list)  # (sid, path, offset, limit) -> [rows]
    for r in reads:
        within[(r["sid"], r["path"], r["read_offset"], r["read_limit"])].append(r)
    # count differing-window re-reads of the same path separately (info, not waste)
    by_path = defaultdict(set)
    for (sid, path, off, lim), rows in within.items():
        by_path[(sid, path)].add((off, lim))
    excl["different_window"] = sum(1 for v in by_path.values() if len(v) > 1)

    for (sid, path, off, lim), rows in within.items():
        for prev, cur in zip(rows, rows[1:]):
            if cur["sidechain"]:
                excl["subagent_context"] += 1
                continue
            between = [ln for ln in edits_by_key[(sid, path)]
                       if prev["line_no"] < ln < cur["line_no"]]
            if between:
                excl["verification_after_edit"] += 1
                continue
            if prev["content_digest"] and cur["content_digest"] \
                    and prev["content_digest"] != cur["content_digest"]:
                excl["file_changed_on_disk"] += 1
                continue
            tok = (cur["content_chars"] or 0) // 4
            m = session_model(con, sid)
            w = usd(tok, m, "cache_w_1h")
            waste_tok += tok
            waste_usd += w
            findings.append({
                "scope": "within_session", "session": sid, "path": path,
                "first_read_line": prev["line_no"], "repeat_line": cur["line_no"],
                "offset": off, "limit": lim, "est_tokens": tok,
                "identical_content": prev["content_digest"] == cur["content_digest"],
                "usd": round(w, 4),
            })

    # cross-session (same project): opportunity, not literal waste
    cross = defaultdict(lambda: {"sessions": {}, "tokens": 0})
    for r in reads:
        c = cross[(r["project"], r["path"])]
        if r["sid"] not in c["sessions"]:
            c["sessions"][r["sid"]] = r
    cross_findings, cross_tok = [], 0
    for (proj, path), c in cross.items():
        if len(c["sessions"]) < 2:
            continue
        sess = sorted(c["sessions"].values(), key=lambda r: r["ts"] or "")
        rep = sess[1:]
        tok = sum((r["content_chars"] or 0) // 4 for r in rep)
        cross_tok += tok
        cross_findings.append({
            "scope": "cross_session", "project": proj, "path": path,
            "sessions_reading": len(sess),
            "sessions": [r["sid"] for r in sess],
            "est_tokens_repeat_reads": tok,
        })
    cross_findings.sort(key=lambda f: -f["sessions_reading"])

    return {
        "category": "D4 repeated file reads", "measurable": True,
        "waste_tokens": waste_tok, "waste_usd": round(waste_usd, 2),
        "confidence": "medium", "estimated": True,
        "caveats": [
            "Token sizes are chars/4 ESTIMATES.",
            "Exclusion (b) compaction boundaries is NOT IMPLEMENTABLE: no "
            "compaction events exist in this data (none observed in Phase 1). "
            "If compaction occurred unrecorded, within-session numbers are "
            "INFLATED by re-reads that were actually necessary.",
            "Repeat = same path AND same offset/limit window. Re-reads of a "
            "different window are counted separately (different_window) and "
            "not charged as waste.",
            "Cross-session re-reads are reported as OPPORTUNITY (a persistent "
            "memory could avoid them), not literal waste — a fresh session has "
            "an empty context and the read is locally necessary. They are NOT "
            "included in waste_tokens/waste_usd.",
            "File changes between sessions are not excluded from the "
            "cross-session list when read windows differ (digest comparison "
            "only works for identical windows) — cross-session counts are an "
            "upper bound.",
        ],
        "stats": {"exclusions": excl,
                  "within_session_waste_events": len(findings),
                  "cross_session_paths": len(cross_findings),
                  "cross_session_est_tokens": cross_tok},
        "findings": findings + cross_findings,
    }


# ============================================================================
# D5 — whole-file reads for partial need
# ============================================================================

def d5_wholefile_reads(con) -> dict:
    reads = con.execute(
        """SELECT session_id sid, line_no, path, content_chars, num_lines,
                  total_lines FROM file_events
           WHERE operation='read' AND read_offset IS NULL
             AND read_limit IS NULL AND content_chars > ?
           ORDER BY session_id, line_no""", (D5_MIN_READ_CHARS,)).fetchall()
    findings, waste_tok, waste_usd = [], 0, 0.0
    for r in reads:
        edit = con.execute(
            """SELECT line_no, patch_lines, patch_hunks FROM file_events
               WHERE session_id=? AND path=? AND operation IN ('edit','write')
                 AND line_no > ? ORDER BY line_no LIMIT 1""",
            (r["sid"], r["path"], r["line_no"])).fetchone()
        if not edit or (edit["patch_lines"] or 0) > D5_SMALL_EDIT_LINES:
            continue
        read_tok = (r["content_chars"] or 0) // 4
        # upper bound: nearly all of the read was unneeded for the edit itself
        m = session_model(con, r["sid"])
        w = usd(read_tok, m, "cache_w_1h")
        waste_tok += read_tok
        waste_usd += w
        findings.append({
            "session": r["sid"], "read_line": r["line_no"], "path": r["path"],
            "read_est_tokens": read_tok, "file_lines": r["total_lines"],
            "edit_line_no": edit["line_no"],
            "edit_lines_touched": edit["patch_lines"],
            "usd_upper_bound": round(w, 4),
        })
    findings.sort(key=lambda f: -f["read_est_tokens"])
    return {
        "category": "D5 whole-file reads for partial need", "measurable": True,
        "waste_tokens": waste_tok, "waste_usd": round(waste_usd, 2),
        "confidence": "low", "estimated": True,
        "caveats": [
            "UPPER BOUND with a strong legitimacy argument against it: editing "
            "a few lines correctly often genuinely requires reading the whole "
            "file (imports, conventions, call sites). The whole read is "
            "counted as waste only because the needed fraction is unknowable "
            "from this data. Treat as ceiling, not measurement.",
            "Token sizes are chars/4 ESTIMATES.",
            "Only the FIRST subsequent edit is inspected; a small first edit "
            "followed by large ones still counts as a finding (inflating).",
            "DOUBLE-COUNT RISK with D3 (if the read was oversized) and D4 (if "
            "it was also a repeat read).",
        ],
        "stats": {"qualifying_reads": len(findings)},
        "findings": findings,
    }


# ============================================================================
# D6 — search thrash
# ============================================================================

def d6_search_thrash(con) -> dict:
    findings, waste_tok, waste_usd = [], 0, 0.0
    n_spans = n_excluded_exploratory = 0
    for s in con.execute("SELECT id FROM sessions"):
        sid = s["id"]
        prompts = con.execute(
            """SELECT line_no, is_exploratory FROM turns
               WHERE session_id=? AND is_human=1 ORDER BY line_no""",
            (sid,)).fetchall()
        calls = con.execute(
            """SELECT tc.line_no, tc.tool_name, tc.id,
                      COALESCE(tr.est_tokens, 0) result_tokens
               FROM tool_calls tc
               LEFT JOIN tool_results tr ON tr.tool_use_id = tc.id
               WHERE tc.session_id=? ORDER BY tc.line_no""", (sid,)).fetchall()
        if not prompts or not calls:
            continue
        bounds = [p["line_no"] for p in prompts] + [float("inf")]
        for i, p in enumerate(prompts):
            span = [c for c in calls if bounds[i] < c["line_no"] < bounds[i + 1]]
            if not span:
                continue
            n_spans += 1
            pre_read = []
            for c in span:
                if c["tool_name"] == "Read":
                    break
                pre_read.append(c)
            run = []
            best_run = []
            for c in pre_read:
                if c["tool_name"] in ("Grep", "Glob"):
                    run.append(c)
                    if len(run) > len(best_run):
                        best_run = list(run)
                else:
                    run = []
            if len(best_run) < D6_RUN_LENGTH:
                continue
            if p["is_exploratory"]:
                n_excluded_exploratory += 1
                continue
            # conservative: only the results of searches beyond the second
            tok = sum(c["result_tokens"] for c in best_run[2:])
            m = session_model(con, sid)
            w = usd(tok, m, "cache_w_1h")
            waste_tok += tok
            waste_usd += w
            findings.append({
                "session": sid, "task_start_line": p["line_no"],
                "consecutive_searches": len(best_run),
                "search_lines": [c["line_no"] for c in best_run],
                "est_tokens_beyond_two": tok, "usd": round(w, 4),
            })
    return {
        "category": "D6 search thrash", "measurable": True,
        "waste_tokens": waste_tok, "waste_usd": round(waste_usd, 2),
        "confidence": "low", "estimated": True,
        "caveats": [
            "'Thrash' is a heuristic: 3+ consecutive Grep/Glob before the first "
            "Read of a task. Iterative narrowing is often competent searching, "
            "not waste. Only search results beyond the second call are counted.",
            "Task = span between human prompts; multi-task prompts blur this.",
            "Exploratory-prompt exclusion depends on a keyword list matched at "
            "ingest; prompts exploratory in intent but not in wording are NOT "
            "excluded (inflating).",
            "Token sizes are chars/4 ESTIMATES.",
        ],
        "stats": {"task_spans_seen": n_spans,
                  "thrash_events": len(findings),
                  "excluded_exploratory": n_excluded_exploratory,
                  "per_10_sessions": round(
                      10 * len(findings) /
                      max(1, con.execute("SELECT COUNT(*) n FROM sessions")
                          .fetchone()["n"]), 2)},
        "findings": findings,
    }


# ============================================================================
# D7 — compaction churn
# ============================================================================

def d7_compaction_churn(con) -> dict:
    n = con.execute(
        "SELECT COUNT(*) n FROM events WHERE kind LIKE '%compact%'").fetchone()["n"]
    return {
        "category": "D7 compaction churn", "measurable": False,
        "waste_tokens": 0, "waste_usd": 0.0, "confidence": "n/a",
        "estimated": False,
        "caveats": [
            "NOT MEASURABLE FROM THIS DATA: zero compaction events found in any "
            "of the 27 sessions (Phase 1 discovery and the events table agree: "
            f"{n} compaction-like events). Either these sessions never hit the "
            "compaction threshold, or Claude Code does not record compaction "
            "in the transcript. Measuring this needs either longer sessions "
            "that visibly compact, or a client version that logs the event.",
        ],
        "stats": {"compaction_events_found": n},
        "findings": [],
    }


# ============================================================================

ALL = [d1_cache_efficiency, d2_unused_tools, d3_oversized_output,
       d4_repeated_reads, d5_wholefile_reads, d6_search_thrash,
       d7_compaction_churn]


def run_all(db_path: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return [d(con) for d in ALL]
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="lattice.db")
    ap.add_argument("--json", default="findings.json",
                    help="write full findings (with evidence) here")
    args = ap.parse_args()

    results = run_all(args.db)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)

    print(f"{'category':<42} {'measurable':<10} {'tokens':>10} {'USD':>8} "
          f"{'conf':<6} findings")
    for r in results:
        print(f"{r['category']:<42} {str(r['measurable']):<10} "
              f"{r['waste_tokens']:>10,} {r['waste_usd']:>8.2f} "
              f"{r['confidence']:<6} {len(r['findings'])}")
    print(f"\nFull evidence written to {args.json}")
    print("Dollar figures are notional API-rate equivalents; this data comes "
          "from a subscription.")


if __name__ == "__main__":
    main()
