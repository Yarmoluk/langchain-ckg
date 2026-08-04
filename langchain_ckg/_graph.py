"""Bundled CKG loader + traversal — pure stdlib, no network, no extra deps.

Domain CSVs ship inside the wheel under ``langchain_ckg/domains/``. Schema:
``ConceptID,ConceptLabel,Dependencies,TaxonomyID,SourceURL,source_content_hash``
where Dependencies is pipe-delimited ``id:TYPE:weight`` entries. Every node
carries the SHA-256 of its source page bytes at extraction time.
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

DOMAINS_DIR = Path(__file__).parent / "domains"

_GRAPH_CACHE: dict = {}


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
        for row in csv.DictReader(f):
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
