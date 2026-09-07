"""Bundled CKG loader + traversal — pure stdlib, no network, no extra deps.

Domain CSVs ship inside the wheel under ``langchain_ckg/domains/``. Schema:
``ConceptID,ConceptLabel,Dependencies,TaxonomyID,SourceURL,source_content_hash``
where Dependencies is pipe-delimited ``id:TYPE:weight`` entries. Every node
carries the SHA-256 of its source page bytes at extraction time.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path

DOMAINS_DIR = Path(__file__).parent / "domains"

_GRAPH_CACHE: dict = {}

# Authoring-metadata leak filter — kept in sync by hand with the same filter in
# ckg-mcp's src/ckg_mcp/graph.py (separate packages, can't share an import).
# See that file's comment for the full rationale. Audited 2026-09-07: bundled
# agent-memory.csv carries the same leaked rows as ckg-mcp's copy.
_SCALAR_LEAK_RE = re.compile(
    r"^\d+(\.\d+)?$|^(?:true|false)$|^\d{4}-\d{2}-\d{2}$", re.IGNORECASE
)


def _leak_dep_id(dep_str: str) -> str:
    return dep_str.split(":")[0] if ":" in dep_str else dep_str


def filter_leaked_metadata_rows(rows: list[dict]) -> list[dict]:
    """Drop authoring-metadata rows (scalar label, zero edges in either direction)."""
    incoming: set = set()
    for row in rows:
        for dep in (row.get("Dependencies") or "").split("|"):
            dep = dep.strip()
            if dep:
                incoming.add(_leak_dep_id(dep))

    def is_leak(row: dict) -> bool:
        label = (row.get("ConceptLabel") or "").strip()
        if not _SCALAR_LEAK_RE.match(label):
            return False
        if (row.get("Dependencies") or "").strip():
            return False
        cid = (row.get("ConceptID") or "").strip()
        return cid not in incoming

    return [row for row in rows if not is_leak(row)]


def available_domains() -> list[str]:
    """Names of the knowledge-graph domains bundled in this package."""
    return sorted(p.stem for p in DOMAINS_DIR.glob("*.csv"))


def _dep_id(dep_str: str) -> str:
    """Extract concept ID from a dependency string — handles '5' and '5:REQUIRES:0.95'."""
    return dep_str.split(":")[0] if ":" in dep_str else dep_str


def load_graph(domain: str):
    """Load a bundled domain into adjacency maps.

    Returns ``(id_to_label, label_to_id, prerequisites, dependents, taxonomy,
    provenance)`` where provenance maps concept ID to
    ``{"source_url": ..., "source_hash": ...}``.
    """
    if domain in _GRAPH_CACHE:
        return _GRAPH_CACHE[domain]

    csv_path = DOMAINS_DIR / f"{domain}.csv"
    if not csv_path.exists():
        raise ValueError(
            f"Domain '{domain}' is not bundled with langchain-ckg. "
            f"Bundled domains: {', '.join(available_domains())}. "
            "For the full 100+ domain library use CKGHostedRetriever "
            "(hosted, 48h free) — see https://graphifymd.com."
        )

    id_to_label: dict = {}
    label_to_id: dict = {}
    prerequisites: dict = defaultdict(list)
    dependents: dict = defaultdict(list)
    taxonomy: dict = {}
    provenance: dict = {}

    with open(csv_path, encoding="utf-8") as f:
        rows = filter_leaked_metadata_rows(list(csv.DictReader(f)))
        for row in rows:
            cid = row["ConceptID"]
            label = row["ConceptLabel"].strip()
            deps = [d.strip() for d in row["Dependencies"].split("|") if d.strip()]
            id_to_label[cid] = label
            label_to_id[label.lower()] = cid
            taxonomy[cid] = row.get("TaxonomyID", "").strip()
            prerequisites[cid] = deps
            for dep in deps:
                dependents[_dep_id(dep)].append(cid)
            src_url = (row.get("SourceURL") or "").strip()
            src_hash = (row.get("source_content_hash") or "").strip()
            if src_url or src_hash:
                provenance[cid] = {"source_url": src_url, "source_hash": src_hash}

    result = id_to_label, label_to_id, prerequisites, dependents, taxonomy, provenance
    _GRAPH_CACHE[domain] = result
    return result


def bfs_subgraph(start_id: str, adj: dict, id_to_label: dict, max_depth: int) -> list[dict]:
    """Breadth-first traversal over declared edges; deterministic, no scoring."""
    visited: set = set()
    queue = deque([(start_id, 0)])
    results = []
    while queue:
        cid, depth = queue.popleft()
        cid = _dep_id(cid)
        if cid in visited or depth > max_depth:
            continue
        visited.add(cid)
        neighbors = adj.get(cid, [])
        results.append({
            "concept": id_to_label.get(cid, cid),
            "related": [id_to_label.get(_dep_id(n), _dep_id(n)) for n in neighbors],
            "depth": depth,
        })
        for n in neighbors:
            n_id = _dep_id(n)
            if n_id not in visited:
                queue.append((n_id, depth + 1))
    return results
