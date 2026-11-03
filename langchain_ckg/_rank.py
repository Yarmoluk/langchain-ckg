"""Ranked concept resolution and lexical domain routing — pure stdlib.

Ported from ckg-mcp's ``src/ckg_mcp/graph.py`` (separate packages, cannot share
an import; kept in sync by hand, the way the leak filter in ``_graph.py`` is).
The ranker replaces plain substring matching: ``"NIM-Microservice"``,
``"nim_microservice"`` and ``"Microservice NIM"`` all resolve to the same node,
a question that merely contains the concept name still finds it, and a typo
within difflib ratio 0.6 still lands. The router picks the bundled domain whose
labels cover a question best, without a model call.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import defaultdict

_EDGE_PUNCT = re.compile(r"""(?<=\s)[("'\[]+|^[("'\[]+|[?!.,;:)"'\]]+(?=\s|$)""")

_STOP = {"the", "of", "and", "a", "an", "for", "to", "in", "nvidia"}
# Words a question is made of. A label made only of these never wins the "inside" or
# "shared" tiers for a question that merely contains them.
_QUESTION_WORDS = {
    "what", "is", "are", "how", "doe", "do", "which", "why", "when", "where", "this", "that",
    "it", "list", "all", "from", "between", "with", "relate", "related", "prerequisite", "chain",
    "explain", "describe", "define",
}


def _norm(text: str) -> str:
    """'NIM-Microservice', 'nim_microservice' and 'Nim Microservice' all compare equal.

    Sentence punctuation at a word's edge is dropped ("Regression?" → "regression") so a
    question passed whole still names its concepts; "node.js", "c++" and "3.5" are kept.
    """
    text = re.sub(r"[\s_\-/]+", " ", text.lower())
    return _EDGE_PUNCT.sub("", text).strip()


def _words(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _norm(text).split()}


def _is_junk_label(label: str) -> bool:
    # Description sentences, filenames and bare scalars that leaked into some CSVs as
    # concept rows. They stay reachable by exact name but never win a fuzzy match.
    return (
        len(label.split()) > 8
        or label.endswith(".md")
        or not re.search(r"[a-z]", label, re.I)
    )


def resolve_concept(label_to_id: dict, query: str) -> tuple[str | None, str, list[str]]:
    """Resolve a concept name to a node, and say how.

    Returns ``(concept_id or None, how, candidate_labels)``. ``how`` is "exact",
    "normalized" (same words, different case/separators/order) or "partial" (every
    query word appears in exactly one label), or "" when unresolved. Ambiguity is
    returned as up to 5 candidates, never resolved silently.
    """
    q = query.lower().strip()
    if not q:
        return None, "", []
    if q in label_to_id:
        return label_to_id[q], "exact", []

    nq, wq = _norm(q), _words(q)
    labels = [l for l in label_to_id if not _is_junk_label(l)]
    same = [l for l in labels if _norm(l) == nq or (wq and _words(l) == wq)]
    if len(same) == 1:
        return label_to_id[same[0]], "normalized", []

    contains = [l for l in labels if wq and wq <= _words(l)]
    if len(contains) == 1 and not same:
        return label_to_id[contains[0]], "partial", []

    inside = [l for l in labels if _words(l) and _words(l) <= wq]
    close = difflib.get_close_matches(nq, [_norm(l) for l in labels], 5, 0.6)
    by_norm = {_norm(l): l for l in labels}
    shared = [l for l in labels if wq & _words(l) - {"nvidia", "the", "of", "and"}]
    ranked: list[str] = []
    for l in same + contains + inside + [by_norm[c] for c in close] + shared:
        if l not in ranked:
            ranked.append(l)
    return None, "", ranked[:5]


def rank_concepts(label_to_id: dict, query: str, limit: int = 20) -> list[tuple[str, str, str]]:
    """Rank concept labels for a query. Returns ``[(label, concept_id, how)]``.

    Tiers, best first: "exact", "normalized" (same words, any case, separators or
    order), "contains" (every query word is in the label), "inside" (the label's words
    all appear in the query), "substring", "close" (difflib ratio >= 0.6, catches
    typos), "shared" (at least one non-stop word in common). Junk rows are reachable
    only by exact text. Within a tier, shorter labels first, except "inside", where the
    label covering more of the query wins.
    """
    q = query.lower().strip()
    if not q:
        return []
    nq, wq = _norm(q), _words(q) - _STOP
    nq_compact = nq.replace(" ", "")
    scored: dict[str, tuple[int, int, str]] = {}

    def put(label: str, tier: int, how: str) -> None:
        cur = scored.get(label)
        if cur is None or tier < cur[0]:
            scored[label] = (tier, -len(label) if tier == 3 else len(label), how)

    for label in label_to_id:
        if label == q:
            put(label, 0, "exact")
    labels = [l for l in label_to_id if not _is_junk_label(l)]
    norms = {l: _norm(l) for l in labels}
    for l in labels:
        nl, wl = norms[l], _words(l) - _STOP
        if nl == nq or (wq and wl == wq):
            put(l, 1, "normalized")
        elif wq and wq <= wl:
            put(l, 2, "contains")
        elif wl and wl <= wq and wl - _QUESTION_WORDS:
            put(l, 3, "inside")
        elif nq in nl or nq_compact in nl.replace(" ", ""):
            put(l, 4, "substring")
    if len(scored) < limit:
        by_norm: dict[str, str] = {}
        for l in labels:
            by_norm.setdefault(norms[l], l)
        for c in difflib.get_close_matches(nq, list(by_norm), limit, 0.6):
            put(by_norm[c], 5, "close")
    if len(scored) < limit and wq:
        for l in labels:
            if (wq & (_words(l) - _STOP)) - _QUESTION_WORDS:
                put(l, 6, "shared")
    ordered = sorted(scored.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    return [(l, label_to_id[l], how) for l, (_, _, how) in ordered[:limit]]


# ---------------------------------------------------------------------------
# Domain routing — which bundled graph holds the concepts a question names.

_ROUTE_STOP = _STOP | {
    "what", "is", "are", "how", "doe", "do", "which", "list", "all", "this", "that", "it", "with",
    "from", "between", "relate", "related", "relationship", "prerequisite", "chain", "concept",
    "knowledge", "graph", "i", "you", "we", "be", "by", "on", "as", "at", "or", "their", "can",
    "why", "when", "where", "explain", "describe", "define", "definition", "mean", "meaning",
    "role", "purpose", "used", "use", "work", "difference",
}
_ROUTE_INDEX: dict | None = None


def _route_index() -> dict:
    """Per-domain label vocabulary over every bundled domain, built once."""
    global _ROUTE_INDEX
    if _ROUTE_INDEX is not None:
        return _ROUTE_INDEX
    from langchain_ckg._graph import available_domains, load_graph

    domains: dict = {}
    df: dict = defaultdict(int)
    for d in available_domains():
        try:
            _, l2i, *_ = load_graph(d)
        except Exception:
            continue
        words: dict = defaultdict(int)
        labels: dict = {}
        for label in l2i:
            if _is_junk_label(label):
                continue
            ws = _words(label) - _STOP
            labels[_norm(label)] = ws
            for w in ws:
                words[w] += 1
        for w in words:
            df[w] += 1
        domains[d] = {"words": dict(words), "labels": labels, "name_words": _words(d) - _STOP}
    _ROUTE_INDEX = {"domains": domains, "df": dict(df), "n": len(domains)}
    return _ROUTE_INDEX


def route_domains(query: str, limit: int = 5) -> list[tuple[str, float, list[str]]]:
    """Rank bundled domains by how much of the query their labels cover.

    Returns ``[(domain, score, matched_labels)]``, best first, empty when no label
    word overlaps. Score = idf²-weighted overlap of the query's words (question-template
    words removed) with the domain's label vocabulary, plus a bonus for every whole
    label (two or more words) the query contains, plus the domain's own name words.
    Pure lexical, deterministic, no model call.
    """
    idx = _route_index()
    wq = _words(query) - _ROUTE_STOP
    if not wq or not idx["domains"]:
        return []
    n = idx["n"]
    df = idx["df"]
    idf = {w: math.log(1 + n / (1 + df.get(w, 0))) ** 2 for w in wq}
    scored: list[tuple[str, float, list[str]]] = []
    for d, info in idx["domains"].items():
        words = info["words"]
        score = 0.0
        for w in wq:
            if w in words:
                score += idf[w] * (1 + math.log(words[w]))
            if w in info["name_words"]:
                score += idf[w]
        matched = []
        for norm_label, ws in info["labels"].items():
            if len(ws) >= 2 and ws <= wq:
                score += sum(idf[w] for w in ws) * len(ws)
                matched.append(norm_label)
        if score > 0:
            matched.sort(key=lambda l: (-len(l), l))
            scored.append((d, round(score, 3), matched[:3]))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:limit]
