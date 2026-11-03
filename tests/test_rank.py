"""Ranked resolution and routing over the bundled domains."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from langchain_ckg import CKGRetriever, rank_concepts, resolve_concept, route_domains  # noqa: E402
from langchain_ckg._graph import load_graph  # noqa: E402


def _l2i(domain="langgraph"):
    return load_graph(domain)[1]


def test_case_separators_and_order_resolve_to_the_same_node():
    l2i = _l2i()
    cid = l2i["checkpointer"]
    for q in ("Checkpointer", "CHECKPOINTER", "check-pointer", "check_pointer"):
        top = rank_concepts(l2i, q, 5)[0]
        assert top[1] == cid, q


def test_typo_lands_via_close_tier():
    l2i = _l2i()
    label, cid, how = rank_concepts(l2i, "checkpoitner", 5)[0]
    assert cid == l2i["checkpointer"] and how == "close"


def test_question_text_finds_the_concept_it_names():
    l2i = _l2i()
    top = rank_concepts(l2i, "how do I persist state across turns with a checkpointer?", 5)[0]
    assert top[1] == l2i["checkpointer"] and top[2] == "inside"


def test_resolve_concept_returns_candidates_not_a_silent_guess():
    l2i = _l2i()
    cid, how, cands = resolve_concept(l2i, "agent")
    assert cid is None and how == "" and cands, "ambiguous query must come back as candidates"


def test_retriever_uses_ranker_and_reports_tier():
    docs = CKGRetriever(depth=1).invoke("Check Pointer")
    assert docs and docs[0].metadata["concept"] == "Checkpointer"
    assert docs[0].metadata["how"] in {"normalized", "substring", "contains"}


def test_auto_domain_routes_to_the_bundled_graph_that_names_the_concepts():
    routed = route_domains("how do I persist state across turns with a checkpointer?", 3)
    assert routed and routed[0][0] == "langgraph"
    docs = CKGRetriever(domain="auto", depth=1).invoke("resources and prompts in MCP")
    assert docs and docs[0].metadata["domain"] == "mcp-protocol"
    assert route_domains("zqxv", 3) == []


def test_perturbation_suite_beats_legacy_on_every_rewrite():
    sys.path.insert(0, str(REPO / "evals"))
    from search_perturbation import run

    r = run(n=200, seed=7)
    for kind, t in r["by_variant"].items():
        assert t["ranker@5"] >= t["legacy@5"], kind
    assert r["by_variant"]["sep"]["ranker@1"] == r["by_variant"]["sep"]["n"]
    assert r["by_variant"]["order"]["ranker@5"] == r["by_variant"]["order"]["n"]
