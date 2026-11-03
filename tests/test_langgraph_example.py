"""The LangGraph StateGraph example: hosted read-only MCP by default, offline with the bundled retriever."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

langgraph = pytest.importorskip("langgraph")


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "langgraph_state_graph", REPO / "examples" / "langgraph_state_graph.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # LangGraph resolves State annotations via the module
    spec.loader.exec_module(mod)
    return mod


class _EchoModel:
    """Stands in for a chat model: returns the prompt it was given."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return prompt


def _local(mod):
    from langchain_ckg import CKGRetriever
    return mod.build_graph(retriever=CKGRetriever(depth=3))


def test_default_is_hosted_read_only_mcp(monkeypatch):
    from langchain_ckg import CKGHostedRetriever
    mod = _load_example()
    monkeypatch.setenv("CKG_LICENSE_KEY", "test-key")
    r = mod.hosted_retriever()
    assert isinstance(r, CKGHostedRetriever)
    assert r.endpoint == "https://ckg-mcp.onrender.com" and r.domain == "langgraph"
    assert r._mcp_body("x")["params"]["name"] == "query_ckg"  # a read tool
    assert r._build_headers()["X-License-Key"] == "test-key"
    monkeypatch.delenv("CKG_LICENSE_KEY")
    with pytest.raises(SystemExit):
        mod.hosted_retriever()


def test_graph_retrieves_prerequisite_chain_offline():
    mod = _load_example()
    out = _local(mod).invoke({"question": "checkpointer"})
    assert out["context"], "bundled langgraph domain should match 'checkpointer'"
    assert "## Prerequisites" in out["context"][0]
    assert out["sources"][0]["concept"]
    assert out["answer"].startswith("Matched concepts: ")


def test_model_only_sees_graph_context():
    mod = _load_example()
    model = _EchoModel()
    from langchain_ckg import CKGRetriever
    out = mod.build_graph(retriever=CKGRetriever(depth=3), model=model).invoke({"question": "checkpointer"})
    assert len(model.prompts) == 1
    prompt = model.prompts[0]
    assert "using only the knowledge-graph context" in prompt
    for block in out["context"]:
        assert block in prompt
    assert out["answer"] == prompt


def test_no_match_does_not_guess():
    mod = _load_example()
    out = _local(mod).invoke({"question": "zqxv-not-a-concept"})
    assert out["context"] == []
    assert "does not guess" in out["answer"]


def test_hosted_envelope_unwraps_provenance():
    from langchain_ckg.retriever import _unwrap_hosted_content
    import json
    env = json.dumps({
        "text": "## NeMo Guardrails  ·  nvidia-nemo\n\nType: MODULE\n"
                "knowledge_source_ref: https://docs.nvidia.com/nemo\n"
                "source_content_hash:  sha256:" + "f" * 64 + "\n",
        "tokens": 189,
    })
    text, meta = _unwrap_hosted_content(env)
    assert text.startswith("## NeMo Guardrails")
    assert meta == {
        "tokens": 189,
        "source_url": "https://docs.nvidia.com/nemo",
        "source_hash": "sha256:" + "f" * 64,
        "concept": "NeMo Guardrails",
    }
    # Plain text (older servers) passes through untouched.
    assert _unwrap_hosted_content("# plain") == ("# plain", {})
