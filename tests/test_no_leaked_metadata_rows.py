"""Regression test for the authoring-metadata-leak bug, mirrored from ckg-mcp's
tests/test_no_leaked_metadata_rows.py (audited 2026-09-07 — see that repo's
served-csv-meta-rows-leak memory note). langchain-ckg bundles a copy of
agent-memory.csv that carried the same 6 leaked rows.
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from langchain_ckg._graph import load_graph, filter_leaked_metadata_rows, DOMAINS_DIR  # noqa: E402

_SCALAR_RE = re.compile(r"^\d+(\.\d+)?$|^(?:true|false)$|^\d{4}-\d{2}-\d{2}$", re.IGNORECASE)


def _leaked_ids_in_raw_csv(rows: list[dict]) -> set:
    incoming = set()
    for row in rows:
        for dep in (row.get("Dependencies") or "").split("|"):
            dep = dep.strip()
            if dep:
                incoming.add(dep.split(":")[0] if ":" in dep else dep)
    leaked = set()
    for row in rows:
        label = (row.get("ConceptLabel") or "").strip()
        cid = (row.get("ConceptID") or "").strip()
        if (
            _SCALAR_RE.match(label)
            and not (row.get("Dependencies") or "").strip()
            and cid not in incoming
        ):
            leaked.add(cid)
    return leaked


def test_bundled_agent_memory_is_clean():
    id_to_label, *_ = load_graph("agent-memory")
    leaked_labels = {"7", "2026-07-06", "false", "0.123", "0.120", "269"}
    assert not (set(id_to_label.values()) & leaked_labels), (
        f"bundled agent-memory.csv still served a leaked metadata row: "
        f"{set(id_to_label.values()) & leaked_labels}"
    )


def test_no_bundled_domain_serves_a_leaked_metadata_row():
    failures = {}
    for path in sorted(DOMAINS_DIR.glob("*.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            raw_rows = list(csv.DictReader(f))
        if not raw_rows or "ConceptLabel" not in raw_rows[0]:
            continue
        filtered = filter_leaked_metadata_rows(raw_rows)
        still_leaked = _leaked_ids_in_raw_csv(filtered)
        if still_leaked:
            failures[path.stem] = still_leaked
    assert not failures, f"bundled domains still contain leaked metadata rows: {failures}"


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
