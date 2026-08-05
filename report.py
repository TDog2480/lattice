"""Phase 4: markdown report over lattice.db + detector findings.

Run: python report.py [--db lattice.db] [--out report.md]

Layout (per the project spec's HOW TO REPORT rules):
  1. Reasons the numbers may be inflated — FIRST, not a footnote
  2. Total spend: four-way token split, dollars, per repo, per session
  3. Waste table (ranked), mechanical vs informational split
  4. Top files by cross-session reads; top 10 most expensive sessions
  5. Kill criteria table
  6. Not-measurable categories

Never prints prompt/file content — paths, counts, timestamps, tokens only.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

from detectors import (CACHE_READ, CACHE_W_1H, CACHE_W_5M, PRICING, rate,
                       run_all)


def session_costs(con) -> list[dict]:
    """Per-session four-way token split and notional USD, priced per call model."""
    out = defaultdict(lambda: {"input": 0, "output": 0, "cw_1h": 0, "cw_5m": 0,
                               "cread": 0, "usd": 0.0, "models": set(),
                               "n_calls": 0})
    rows = con.execute(
        """SELECT session_id sid, model, input_tokens i, output_tokens o,
                  cache_creation cc, cache_read cr, eph_1h, eph_5m
           FROM usage WHERE model != '<synthetic>'""").fetchall()
    for r in rows:
        s = out[r["sid"]]
        rt = rate(r["model"])
        i, o, cr = r["i"] or 0, r["o"] or 0, r["cr"] or 0
        e1, e5 = r["eph_1h"] or 0, r["eph_5m"] or 0
        cc = r["cc"] or 0
        if e1 + e5 == 0 and cc:      # split unrecorded: price at 1h (observed norm)
            e1 = cc
        s["input"] += i
        s["output"] += o
        s["cw_1h"] += e1
        s["cw_5m"] += e5
        s["cread"] += cr
        s["usd"] += (i * rt["input"] + o * rt["output"]
                     + e1 * rt["input"] * CACHE_W_1H
                     + e5 * rt["input"] * CACHE_W_5M
                     + cr * rt["input"] * CACHE_READ) / 1_000_000
        s["models"].add(r["model"] or "?")
        s["n_calls"] += 1
    for sid, s in out.items():
        meta = con.execute(
            "SELECT project, first_ts, last_ts FROM sessions WHERE id=?",
            (sid,)).fetchone()
        s.update(sid=sid, project=meta["project"], first_ts=meta["first_ts"])
    return sorted(out.values(), key=lambda s: -s["usd"])


def session_reason(s: dict, d1_by_session: dict) -> str:
    """One-line 'why was this expensive'."""
    rt = rate(sorted(s["models"])[0])
    comp = {
        "cache writes": (s["cw_1h"] * CACHE_W_1H + s["cw_5m"] * CACHE_W_5M) * rt["input"],
        "cache reads": s["cread"] * CACHE_READ * rt["input"],
        "output": s["output"] * rt["output"],
        "uncached input": s["input"] * rt["input"],
    }
    top = max(comp, key=comp.get)
    pct = 100 * comp[top] / max(1e-9, sum(comp.values()))
    bits = [f"{pct:.0f}% of cost is {top} ({s['n_calls']} API calls)"]
    inv = d1_by_session.get(s["sid"], [])
    if inv:
        biggest = max(f["rewritten_tokens"] for f in inv)
        reasons = {f["labeled_reason"] for f in inv if f["labeled_reason"]}
        bits.append(f"{len(inv)} cache invalidation(s), largest {biggest:,} tok"
                    + (f" [{', '.join(sorted(reasons))}]" if reasons else ""))
    if len(s["models"]) > 1:
        bits.append(f"model switch mid-session: {', '.join(sorted(s['models']))}")
    return "; ".join(bits)


def fmt_tok(n: int) -> str:
    return f"{n:,}"


def build(db_path: str) -> str:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    det = {d["category"].split(" ")[0]: d for d in run_all(db_path)}
    sess = session_costs(con)
    total_usd = sum(s["usd"] for s in sess)
    tot = {k: sum(s[k] for s in sess) for k in
           ("input", "output", "cw_1h", "cw_5m", "cread")}
    d1_by_session = defaultdict(list)
    for f in det["D1"]["findings"]:
        d1_by_session[f["session"]].append(f)

    n_total = con.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"]
    date_range = (min(s["first_ts"] or "?" for s in sess)[:10],
                  max(s["first_ts"] or "?" for s in sess)[:10])

    L = []
    A = L.append
    A("# Claude Code token waste report")
    A("")
    A(f"Dataset: **{n_total} sessions ({len(sess)} with billable API calls)**, "
      f"{date_range[0]} to {date_range[1]}, "
      f"{len({s['project'] for s in sess})} projects, one user. "
      "Source: local Claude Code JSONL transcripts (read-only).")
    A("")
    A("> **All dollar figures are notional.** This usage is billed through a "
      "Claude Code subscription, not metered API pricing. Dollars answer "
      "\"what would this cost at current API rates\" (Sonnet 5 at its "
      "introductory $2/$10 per MTok) — nothing more.")

    # ------------------------------------------------------------------ 1
    A("")
    A("## 1. Read this first: why these numbers may overstate waste")
    A("")
    A("- **Sample is small and skewed.** 27 sessions from one user over ~6 "
      "weeks; 81% of sessions come from two research repos. Nothing here "
      "generalizes beyond this machine without more data.")
    A("- **Most token sizes are estimates.** The transcript records no token "
      "counts for tool results or file reads; D3-D6 use chars/4. Only D1 and "
      "the spend totals use API-measured token counts.")
    A("- **Categories overlap.** A large repeated whole-file read could appear "
      "in D3, D4 and D5 simultaneously. Do not sum categories without dedup "
      "(in this dataset the issue is moot - D3 and D5 found nothing).")
    A("- **Some \"waste\" is defensible.** D1's largest events are caused by "
      "mid-session model switches - the cache is model-scoped, so the rewrite "
      "is the unavoidable price of switching, not a bug. Whether to call it "
      "waste depends on whether the switch was worth it.")
    A("- **Missing filters inflate what remains.** Compaction boundaries are "
      "not recorded, so the D4 compaction exclusion could not be applied "
      "(no compaction was observed either, so the risk is low here).")
    A("- **And the headline finding cuts the other way:** five of seven "
      "detectors found zero or near-zero waste. Claude Code's harness already "
      "caps tool output (D3=0), the agent did not thrash searches (D6=0), did "
      "not re-read files without cause within a session (D4 within-session=0 "
      "after exclusions), and cache hit rates are excellent (median 93%). "
      "**On this dataset, the strong version of the waste thesis does not "
      "hold.** What remains is one mechanical bug class (cache invalidation, "
      "mostly self-inflicted) and one opportunity (cross-session memory).")

    # ------------------------------------------------------------------ 2
    A("")
    A("## 2. Total spend")
    A("")
    A("| token class | tokens | notional USD |")
    A("|---|---:|---:|")
    # price aggregate per class using per-session model mix is already folded
    # into usd; recompute class dollars for display with a weighted approach:
    class_usd = {"input": 0.0, "output": 0.0, "cw": 0.0, "cread": 0.0}
    for s in sess:
        rt = rate(sorted(s["models"])[0])
        class_usd["input"] += s["input"] * rt["input"] / 1e6
        class_usd["output"] += s["output"] * rt["output"] / 1e6
        class_usd["cw"] += (s["cw_1h"] * CACHE_W_1H + s["cw_5m"] * CACHE_W_5M) \
            * rt["input"] / 1e6
        class_usd["cread"] += s["cread"] * CACHE_READ * rt["input"] / 1e6
    A(f"| uncached input | {fmt_tok(tot['input'])} | ${class_usd['input']:.2f} |")
    A(f"| cache create (1h: {fmt_tok(tot['cw_1h'])}, 5m: {fmt_tok(tot['cw_5m'])}) "
      f"| {fmt_tok(tot['cw_1h'] + tot['cw_5m'])} | ${class_usd['cw']:.2f} |")
    A(f"| cache read | {fmt_tok(tot['cread'])} | ${class_usd['cread']:.2f} |")
    A(f"| output | {fmt_tok(tot['output'])} | ${class_usd['output']:.2f} |")
    A(f"| **total** | {fmt_tok(sum(tot.values()))} | **${total_usd:.2f}** |")
    A("")
    A("Uncached input is 0.01% of input tokens - prompt caching is working "
      "almost perfectly at the request level.")
    A("")
    A("### Per repo")
    A("")
    A("| repo | sessions | notional USD |")
    A("|---|---:|---:|")
    per_repo = defaultdict(lambda: [0, 0.0])
    for s in sess:
        per_repo[s["project"]][0] += 1
        per_repo[s["project"]][1] += s["usd"]
    for proj, (n, cost) in sorted(per_repo.items(), key=lambda kv: -kv[1][1]):
        A(f"| {proj} | {n} | ${cost:.2f} |")

    # ------------------------------------------------------------------ 3
    A("")
    A("## 3. Waste by category")
    A("")
    A("| category | class | tokens | notional USD | % of total | confidence | basis |")
    A("|---|---|---:|---:|---:|---|---|")
    klass = {"D1": "MECHANICAL", "D2": "MECHANICAL", "D3": "MECHANICAL",
             "D4": "INFORMATIONAL", "D5": "INFORMATIONAL", "D6": "INFORMATIONAL"}
    ranked = sorted((d for k, d in det.items() if k != "D7"),
                    key=lambda d: -d["waste_usd"])
    for d in ranked:
        k = d["category"].split(" ")[0]
        basis = "not measurable" if not d["measurable"] else \
            ("estimated (chars/4)" if d["estimated"] else "measured")
        pct = 100 * d["waste_usd"] / total_usd if total_usd else 0
        A(f"| {d['category']} | {klass[k]} | {fmt_tok(d['waste_tokens'])} "
          f"| ${d['waste_usd']:.2f} | {pct:.1f}% | {d['confidence']} | {basis} |")
    mech = sum(d["waste_usd"] for k, d in det.items() if k in ("D1", "D2", "D3"))
    info = sum(d["waste_usd"] for k, d in det.items() if k in ("D4", "D5", "D6"))
    A(f"| **mechanical total** | | | **${mech:.2f}** "
      f"| {100 * mech / total_usd:.1f}% | | |")
    A(f"| **informational total** | | | **${info:.2f}** "
      f"| {100 * info / total_usd:.1f}% | | |")
    A("")
    cross_tok = det["D4"]["stats"]["cross_session_est_tokens"]
    A(f"Cross-session re-reads (~{fmt_tok(cross_tok)} est. tokens across "
      f"{det['D4']['stats']['cross_session_paths']} files) are informational "
      "OPPORTUNITY - a fresh session must re-read, so it is excluded from the "
      "waste totals above. It is what persistent cross-session memory could "
      "save.")
    A("")
    A("D1 detail: mid-session cache invalidations by API-labeled cause: "
      + ", ".join(f"{k}={v}" for k, v in
                  det["D1"]["stats"]["api_labeled_miss_reasons"].items())
      + f". Cache hit rate distribution across sessions: "
        f"{det['D1']['stats']['hit_rate_distribution']}.")

    # ------------------------------------------------------------------ 4
    A("")
    A("## 4. Top files by cross-session read count")
    A("")
    A("| # | path | sessions reading it | est. repeat-read tokens |")
    A("|---:|---|---:|---:|")
    cross = [f for f in det["D4"]["findings"] if f.get("scope") == "cross_session"]
    for i, f in enumerate(cross[:15], 1):
        A(f"| {i} | {f['path']} | {f['sessions_reading']} "
          f"| {fmt_tok(f['est_tokens_repeat_reads'])} |")
    if not cross:
        A("| - | (none) | | |")

    # ------------------------------------------------------------------ 5
    A("")
    A("## 5. Top 10 most expensive sessions")
    A("")
    A("| session | repo | notional USD | why |")
    A("|---|---|---:|---|")
    for s in sess[:10]:
        A(f"| {s['sid'][:8]} | {s['project'].split('-')[-1]} | ${s['usd']:.2f} "
          f"| {session_reason(s, d1_by_session)} |")

    # ------------------------------------------------------------------ 6
    A("")
    A("## 6. Kill criteria")
    A("")
    top3 = sum(d["waste_usd"] for d in ranked[:3])
    top3_pct = 100 * top3 / total_usd if total_usd else 0
    hit = det["D1"]["stats"]["hit_rate_distribution"].get("median", 0) * 100
    n_cross = det["D4"]["stats"]["cross_session_paths"]
    thrash10 = det["D6"]["stats"]["per_10_sessions"]
    mcp_unused = max((f["mcp_never_invoked"] for f in det["D2"]["findings"]),
                     default=0)
    info_share = 100 * info / max(1e-9, mech + info)
    info_share_with_opp = None
    cross_usd = cross_tok * rate("claude-sonnet-5")["input"] * CACHE_W_1H / 1e6
    info_share_with_opp = 100 * (info + cross_usd) / max(1e-9, mech + info + cross_usd)

    def row(name, value, threshold, passed, note=""):
        A(f"| {name} | {value} | {threshold} | "
          f"{'**PASS**' if passed else 'FAIL'} | {note} |")

    A("| measurement | value | threshold | pass/fail | note |")
    A("|---|---|---|---|---|")
    row("Top 3 waste categories, % of total spend", f"{top3_pct:.1f}%",
        ">=30%", top3_pct >= 30)
    row("Cache hit rate (median per session)", f"{hit:.1f}%", "<70%",
        hit < 70, "caching works; thesis needed it broken")
    row("Files reread across sessions", str(n_cross), ">=10", n_cross >= 10,
        "opportunity, not measured waste")
    row("Search thrash per 10 sessions", str(thrash10), ">=3", thrash10 >= 3)
    row("MCP tools loaded, never called", str(mcp_unused), ">=1",
        mcp_unused >= 1, "count only - token cost NOT measurable, and this "
        "client defers schemas, so the cost is likely ~0")
    row("Informational share of waste", f"{info_share:.1f}%", ">=25%",
        info_share >= 25,
        f"{info_share_with_opp:.0f}% if cross-session opportunity is counted")
    A("")
    n_pass = sum([top3_pct >= 30, hit < 70, n_cross >= 10, thrash10 >= 3,
                  mcp_unused >= 1, info_share >= 25])
    A(f"**{n_pass} of 6 criteria pass.** The two that pass cleanly "
      "(cross-session rereads, unused MCP tools) are the two whose token cost "
      "this data cannot or does not measure. The criteria that required "
      "demonstrating measured, in-context waste (cache broken, search thrash, "
      "dominant waste share) all fail.")

    # ------------------------------------------------------------------ 7
    A("")
    A("## 7. Not measurable from this data")
    A("")
    for k in ("D2", "D7"):
        A(f"**{det[k]['category']}**")
        for c in det[k]["caveats"]:
            A(f"- {c}")
        A("")

    A("## 8. Bottom line")
    A("")
    A(f"Total notional spend: **${total_usd:.2f}**. Measured waste: "
      f"**${mech + info:.2f} ({100 * (mech + info) / total_usd:.1f}%)**, "
      "nearly all of it D1 cache invalidations - and the largest of those are "
      "the model-scoped cost of switching models mid-session, which is "
      "arguably a user choice, not agent waste. There is not enough measured "
      "waste in this dataset to support the strong thesis. The defensible "
      "residual claims are narrower: (a) mid-session cache invalidation is "
      "real, diagnosable, and worth surfacing to users; (b) cross-session "
      "file re-reading is a real memory opportunity (~"
      f"{fmt_tok(cross_tok)} est. tokens here); (c) two categories the thesis "
      "leans on (tool schema overhead, compaction churn) cannot be measured "
      "from transcripts at all - deciding on them requires request-payload "
      "capture or longer sessions. More data (more users, bigger repos, "
      "longer sessions) could change this picture; this dataset alone argues "
      "against building.")
    con.close()
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="lattice.db")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()
    md = build(args.db)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {args.out} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
