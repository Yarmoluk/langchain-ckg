"""Legacy substring search vs the ranked resolver on the bundled domains.

Mirrors ckg-mcp's evals/run.py search suite: real multi-word labels rewritten the ways
callers rewrite them (case, separators, word order, last word dropped, one typo), scored
as hit@1 / hit@5 for the legacy substring search and for ``rank_concepts``. A junk row
anywhere in a result list counts as a miss.

    python evals/search_perturbation.py [--n 400] [--seed 7] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ckg._graph import available_domains, load_graph  # noqa: E402
from langchain_ckg._rank import _is_junk_label, rank_concepts  # noqa: E402


def legacy_search(label_to_id: dict, query: str) -> list[str]:
    """CKGRetriever matching as shipped through 0.7.0: substring, then any-word."""
    q = query.lower().strip()
    hits = [l for l in label_to_id if q in l]
    if not hits:
        words = q.split()
        hits = [l for l in label_to_id if any(w in l for w in words)]
    return hits[:20]


def variants(label: str) -> list[tuple[str, str]]:
    w = re.split(r"[\s_\-/]+", label)
    out = [("case", label.upper()), ("sep", "_".join(w) if " " in label else " ".join(w))]
    if len(w) in (2, 3):
        out.append(("order", " ".join(reversed(w))))
    if len(w) >= 3:
        out.append(("drop", " ".join(w[:-1])))
    longest = max(range(len(w)), key=lambda i: len(w[i]))
    if len(w[longest]) >= 5:
        t = w[longest]
        w2 = list(w)
        w2[longest] = t[:2] + t[3] + t[2] + t[4:]
        out.append(("typo", " ".join(w2)))
    return out


def run(n: int = 400, seed: int = 7) -> dict:
    rng = random.Random(seed)
    pool = []
    for d in available_domains():
        _, l2i, *_ = load_graph(d)
        for label in l2i:
            if not _is_junk_label(label) and len(label.split()) >= 2 and re.search(r"[a-z]{3}", label):
                pool.append((d, label, l2i))
    sample = rng.sample(pool, min(n, len(pool)))
    tally: dict = {}
    for _, label, l2i in sample:
        for kind, q in variants(label):
            old = legacy_search(l2i, q)
            now = [l for l, _, _ in rank_concepts(l2i, q, 20)]
            t = tally.setdefault(kind, {"n": 0, "legacy@1": 0, "legacy@5": 0, "ranker@1": 0, "ranker@5": 0})
            t["n"] += 1
            for name, lst in (("legacy", old), ("ranker", now)):
                clean = not any(_is_junk_label(l) for l in lst)
                t[f"{name}@1"] += clean and bool(lst) and lst[0] == label
                t[f"{name}@5"] += clean and label in lst[:5]
    return {"domains": len(available_domains()), "labels": len(sample), "pool": len(pool), "by_variant": tally}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = run(a.n, a.seed)
    print(f"**bundled domains** · {r['domains']} graphs · {r['labels']} labels sampled of {r['pool']}\n")
    print("| rewrite | n | legacy hit@1 | legacy hit@5 | ranker hit@1 | ranker hit@5 |")
    print("|---|---:|---:|---:|---:|---:|")
    for kind in ("case", "sep", "order", "drop", "typo"):
        t = r["by_variant"].get(kind)
        if not t:
            continue
        pct = lambda k: f"{100 * t[k] / t['n']:.1f}%"
        print(f"| {kind} | {t['n']} | {pct('legacy@1')} | {pct('legacy@5')} | {pct('ranker@1')} | {pct('ranker@5')} |")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
