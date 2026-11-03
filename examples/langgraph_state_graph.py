"""Example 3 — a LangGraph StateGraph with a CKG retrieval node over MCP.

The graph has two nodes. ``retrieve`` queries a hosted CKG MCP server with
:class:`langchain_ckg.CKGHostedRetriever` and writes the prerequisite chain
plus its source provenance into state. ``answer`` turns that context into
prose. The model only phrases; the facts come from declared edges, and every
concept in the context carries a ``source_url`` and ``source_hash`` the caller
can verify.

The MCP path is read-only by construction: the server exposes query tools
only (``query_ckg``, ``search_concepts``, ``get_prerequisites`` ...), nothing
that writes, and no graph data ships with this example. Get a free 100-call
trial key at https://tally.so/r/b5Ep4g and export it as ``CKG_LICENSE_KEY``.

    pip install langchain-ckg[langgraph]
    export CKG_LICENSE_KEY=...                     # X-License-Key for the server
    python examples/langgraph_state_graph.py checkpointer

Without ``ANTHROPIC_API_KEY`` the ``answer`` node is a deterministic formatter.
``--local`` swaps in the in-process ``CKGRetriever`` over the graphs bundled in
this package, which is what the offline tests use.
"""

import os
import sys
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from langchain_ckg import CKGHostedRetriever, CKGRetriever

ENDPOINT = "https://ckg-mcp.onrender.com"
DOMAIN = "langgraph"


class State(TypedDict, total=False):
    question: str
    context: List[str]  # one rendered subgraph per matched concept
    sources: List[dict]  # {"concept", "source_url", "source_hash"} per concept
    answer: str


def hosted_retriever(license_key: Optional[str] = None) -> CKGHostedRetriever:
    """Read-only MCP client for the hosted ``langgraph`` domain."""
    key = license_key or os.environ.get("CKG_LICENSE_KEY", "")
    if not key:
        raise SystemExit(
            "CKG_LICENSE_KEY is not set. Free 100-call trial key: "
            "https://tally.so/r/b5Ep4g (or run with --local for the bundled graph)."
        )
    return CKGHostedRetriever(endpoint=ENDPOINT, domain=DOMAIN, license_key=key, depth=3)


def build_graph(retriever=None, model=None):
    """Compile the two-node graph.

    Args:
        retriever: any LangChain retriever. Default: :func:`hosted_retriever`,
            the read-only MCP client. Pass ``CKGRetriever()`` for in-process.
        model: any LangChain chat model with ``.invoke(str)``; ``None`` means the
            answer node formats the retrieved context without a model call.
    """
    retriever = retriever or hosted_retriever()

    def retrieve(state: State) -> State:
        docs = retriever.invoke(state["question"])
        return {
            "context": [d.page_content for d in docs],
            "sources": [
                {
                    "concept": d.metadata.get("concept", ""),
                    "source_url": d.metadata.get("source_url", ""),
                    "source_hash": d.metadata.get("source_hash", ""),
                }
                for d in docs
            ],
        }

    def answer(state: State) -> State:
        if not state.get("context"):
            return {"answer": "No matching concept in the graph; the graph does not guess."}
        if model is None:
            head = [s["concept"] for s in state["sources"]]
            return {
                "answer": "Matched concepts: "
                + ", ".join(head)
                + ". The prerequisite chains above are the answer; "
                "verify any node with `curl -s <source_url> | sha256sum`."
            }
        prompt = (
            "Answer the question using only the knowledge-graph context. "
            "Cite concept names as they appear. If the context does not cover "
            "the question, say so.\n\n"
            f"Question: {state['question']}\n\nContext:\n" + "\n\n".join(state["context"])
        )
        result = model.invoke(prompt)
        return {"answer": getattr(result, "content", str(result))}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def _default_model():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None
    return ChatAnthropic(model="claude-sonnet-5-5", max_tokens=600)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--local"]
    local = "--local" in sys.argv
    question = " ".join(args) or "checkpointer"
    retriever = CKGRetriever(depth=3) if local else hosted_retriever()
    app = build_graph(retriever=retriever, model=_default_model())
    out = app.invoke({"question": question})
    for block in out["context"]:
        print(block)
        print("-" * 60)
    print(out["answer"])
    print()
    for s in out["sources"]:
        print(f"{s['concept']}: {s['source_url']}  {s['source_hash']}")
