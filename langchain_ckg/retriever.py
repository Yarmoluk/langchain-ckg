"""
CKGRetriever — LangChain BaseRetriever backed by a Compact Knowledge Graph.

Usage:
    from langchain_ckg import CKGRetriever, PolarUsageCallback
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA

    cb = PolarUsageCallback(api_key="sk_...", external_customer_id="license_key_abc")
    retriever = CKGRetriever(domain="nvidia-nemoclaw", depth=3)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever, callbacks=[cb])
    result = qa.invoke("How does NemoClaw handle CUDA kernel fusion?")
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ckg_mcp.graph import load_graph, bfs_subgraph


class CKGRetriever(BaseRetriever):
    """Retrieve structured knowledge from a CKG domain as LangChain Documents.

    Each document contains the prerequisite + dependent subgraph for a matched
    concept, with SHA-256 source provenance in metadata when available.

    Args:
        domain: CKG domain name (e.g. "nvidia-nemoclaw", "nvidia-ai").
                Run `uvx ckg-mcp list_domains` to see all 97 available domains.
        depth: Upstream prerequisite hops to traverse (1–5, default 3).
        k: Maximum number of concept matches to return (default 5).
    """

    domain: str
    depth: int = Field(default=3, ge=1, le=5)
    k: int = Field(default=5, ge=1, le=20)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        id_to_label, label_to_id, prerequisites, dependents, taxonomy = load_graph(self.domain)

        q = query.lower().strip()
        matches = [(label, cid) for label, cid in label_to_id.items() if q in label][:self.k]

        if not matches:
            # Fuzzy: any word in query matches any word in label
            words = q.split()
            matches = [
                (label, cid)
                for label, cid in label_to_id.items()
                if any(w in label for w in words)
            ][:self.k]

        docs = []
        for label, cid in matches:
            prereqs = bfs_subgraph(cid, prerequisites, id_to_label, self.depth)
            deps = bfs_subgraph(cid, dependents, id_to_label, 2)

            lines = [f"# {id_to_label[cid]} ({self.domain})", ""]
            lines.append("## Prerequisites")
            for node in prereqs[1:]:
                lines.append("  " * node["depth"] + f"- {node['concept']}")
            lines.append("")
            lines.append("## Builds toward")
            for node in deps[1:]:
                lines.append("  " * node["depth"] + f"- {node['concept']}")

            tax = taxonomy.get(cid, "")
            if tax:
                lines.append(f"\nTaxonomy: {tax}")

            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "domain": self.domain,
                        "concept": id_to_label[cid],
                        "taxonomy": tax,
                        "source": "ckg-mcp",
                        "provenance": "sha256-anchored",
                    },
                )
            )

        return docs


class CKGHostedRetriever(BaseRetriever):
    """Retrieve from a hosted CKG MCP server (rate-gated, upgradeable).

    Hits ckg-nvidia-ai.onrender.com or any hosted CKG endpoint.
    Free tier: 50 calls/day. Upgrade at https://graphifymd.com/pricing.

    Args:
        endpoint: Base URL of the hosted CKG server.
        domain: CKG domain name (e.g. "nvidia-nemoclaw").
        license_key: Optional Polar license key for unlimited access.
        depth: Prerequisite traversal depth (default 3).
        k: Max concept matches (default 5).
    """

    endpoint: str = "https://ckg-nvidia-ai.onrender.com"
    domain: str = "nvidia-nemoclaw"
    license_key: str = ""
    depth: int = Field(default=3, ge=1, le=5)
    k: int = Field(default=5, ge=1, le=20)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for CKGHostedRetriever: pip install httpx")

        headers = {}
        if self.license_key:
            headers["X-License-Key"] = self.license_key

        try:
            r = httpx.post(
                f"{self.endpoint}/mcp",
                headers=headers,
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "query_ckg",
                        "arguments": {"domain": self.domain, "concept": query, "depth": self.depth},
                    },
                },
                timeout=10,
            )
            if r.status_code == 402:
                data = r.json()
                raise RuntimeError(data.get("error", "Rate limit reached. Upgrade at https://graphifymd.com/pricing"))
            r.raise_for_status()
            content = r.json().get("result", {}).get("content", [{}])[0].get("text", "")
        except RuntimeError:
            raise
        except Exception as e:
            return [Document(page_content=f"CKG query failed: {e}", metadata={"domain": self.domain})]

        return [
            Document(
                page_content=content,
                metadata={"domain": self.domain, "concept": query, "source": self.endpoint},
            )
        ]


class PolarUsageCallback(BaseCallbackHandler):
    """Emit a Polar meter event on every CKG retrieval for metered billing.

    Pass as a callback to any LangChain chain or retriever that wraps CKGRetriever
    or CKGHostedRetriever. Fires a background thread so it never blocks the chain.

    Args:
        api_key: Polar API secret key (sk_...).
        external_customer_id: Customer identifier — license key, user ID, or email.
        meter_name: Polar meter slug to increment (default: "ckg_query").
        cost_per_call: Usage units per retrieval (default: 1).
    """

    def __init__(
        self,
        api_key: str,
        external_customer_id: str,
        meter_name: str = "ckg_query",
        cost_per_call: int = 1,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._external_customer_id = external_customer_id
        self._meter_name = meter_name
        self._cost_per_call = cost_per_call

    def on_retriever_end(
        self,
        documents: List[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not documents:
            return
        domain = documents[0].metadata.get("domain", "unknown")
        threading.Thread(target=self._fire, args=(domain,), daemon=True).start()

    def _fire(self, domain: str) -> None:
        try:
            import httpx
            httpx.post(
                "https://api.polar.sh/v1/billing/meter-events",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "name": self._meter_name,
                    "external_customer_id": self._external_customer_id,
                    "metadata": {"_cost": self._cost_per_call, "domain": domain},
                },
                timeout=5.0,
            )
        except Exception:
            pass
