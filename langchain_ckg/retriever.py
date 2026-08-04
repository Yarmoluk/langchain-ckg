"""
CKGRetriever — LangChain BaseRetriever backed by a Compressed Knowledge Graph.

Usage (langchain >= 1.0 idiom):
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_ckg import CKGRetriever

    retriever = CKGRetriever()  # bundled langgraph domain, offline

    @tool
    def langgraph_map(query: str) -> str:
        '''Look up LangGraph concepts, their prerequisites, and source docs.'''
        return "\\n\\n".join(d.page_content for d in retriever.invoke(query))

    agent = create_agent(model="claude-opus-5", tools=[langgraph_map])
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr


class CKGRetriever(BaseRetriever):
    """Retrieve structured knowledge from a bundled CKG domain — offline, in-process.

    Eleven agent-stack knowledge graphs ship inside this package (LangGraph, MCP,
    A2A, agent memory, agent loop patterns, AWS Bedrock AgentCore, Palantir
    Foundry, Google Gemini agent platform, Microsoft agent stack, CrewAI,
    LlamaIndex) — call :func:`langchain_ckg.available_domains` to list them.
    No network, no API key, no rate limit.

    Each document contains the prerequisite + dependent subgraph for a matched
    concept. Nodes carry SHA-256 source provenance: the matched concept's
    ``source_url`` and ``source_hash`` land in ``Document.metadata``, so any
    claim is verifiable with ``curl -s <source_url> | sha256sum``.

    Domains beyond the bundled set resolve through the optional ``ckg_mcp``
    runtime if installed, or use ``CKGHostedRetriever`` for the full hosted
    library.

    Args:
        domain: Bundled domain name (default "langgraph").
        depth: Upstream prerequisite hops to traverse (1–5, default 3).
        k: Maximum number of concept matches to return (default 5).
    """

    domain: str = "langgraph"
    depth: int = Field(default=3, ge=1, le=5)
    k: int = Field(default=5, ge=1, le=20)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        from langchain_ckg._graph import DOMAINS_DIR, available_domains

        if (DOMAINS_DIR / f"{self.domain}.csv").exists():
            from langchain_ckg._graph import load_graph, bfs_subgraph
        else:
            try:
                from ckg_mcp.graph import load_graph, bfs_subgraph
            except ImportError:
                raise ValueError(
                    f"Domain '{self.domain}' is not bundled with langchain-ckg. "
                    f"Bundled domains: {', '.join(available_domains())}. "
                    "For the full 100+ domain library use CKGHostedRetriever "
                    "(hosted, 48h free) — see https://graphifymd.com."
                )

        graph = load_graph(self.domain)
        id_to_label, label_to_id, prerequisites, dependents, taxonomy = graph[:5]
        provenance = graph[5] if len(graph) > 5 else {}

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

            prov = provenance.get(cid, {})
            if prov.get("source_url"):
                lines.append(f"Source: {prov['source_url']}")
            if prov.get("source_hash"):
                lines.append(f"Source hash: {prov['source_hash']}")

            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "domain": self.domain,
                        "concept": id_to_label[cid],
                        "taxonomy": tax,
                        "source_url": prov.get("source_url", ""),
                        "source_hash": prov.get("source_hash", ""),
                        "provenance": (
                            "sha256-anchored"
                            if prov.get("source_url", "").startswith("http")
                            else "extraction-internal" if prov else "none"
                        ),
                    },
                )
            )

        return docs


class CKGHostedRetriever(BaseRetriever):
    """Retrieve from a hosted CKG MCP server (rate-gated, upgradeable).

    Supports three payment rails — all autonomous, no human required:

    1. x402 (USDC on Base L2): pass ``x402_private_key`` — on 402, signs a
       $0.001 USDC payment and retries automatically.
    2. Lightning: pass ``lightning_invoice_id`` from ``GET /lightning/invoice``
       on the endpoint — included as ``X-Lightning-Invoice-Id`` header.
    3. License key: pass ``license_key`` (Polar) for unlimited annual access.

    Args:
        endpoint: Base URL of the hosted CKG server.
        domain: CKG domain name (e.g. "nvidia-nemo"). Call the server's
            ``list_domains`` tool to see what the endpoint hosts.
        license_key: Polar license key for unlimited access.
        x402_private_key: EVM private key (hex, 0x-prefixed). Enables
            autonomous USDC payment on Base L2 when a 402 is received.
        lightning_invoice_id: Pre-paid Lightning invoice ID from
            ``GET {endpoint}/lightning/invoice``.
        depth: Prerequisite traversal depth (default 3).
        k: Max concept matches (default 5).
    """

    endpoint: str = "https://ckg-nvidia-ai.onrender.com"
    domain: str = "nvidia-nemo"
    license_key: str = ""
    x402_private_key: str = ""
    lightning_invoice_id: str = ""
    depth: int = Field(default=3, ge=1, le=5)
    k: int = Field(default=5, ge=1, le=20)

    _session_id: str = PrivateAttr(default="")

    def _build_headers(self) -> dict:
        headers: dict = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.license_key:
            headers["X-License-Key"] = self.license_key
        if self.lightning_invoice_id:
            headers["X-Lightning-Invoice-Id"] = self.lightning_invoice_id
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _mcp_body(self, query: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_ckg",
                "arguments": {"domain": self.domain, "concept": query, "depth": self.depth},
            },
        }

    def _ensure_session(self, client) -> None:
        """Open an MCP streamable-HTTP session: initialize + initialized notify."""
        if self._session_id:
            return
        r = client.post(
            f"{self.endpoint}/mcp",
            headers=self._build_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "langchain-ckg", "version": "0.6.0"},
                },
            },
        )
        r.raise_for_status()
        self._session_id = r.headers.get("mcp-session-id", "")
        client.post(
            f"{self.endpoint}/mcp",
            headers=self._build_headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    @staticmethod
    def _parse_rpc(r) -> dict:
        """Parse a JSON-RPC response that may arrive as plain JSON or SSE."""
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    import json
                    msg = json.loads(line[5:].strip())
                    if "result" in msg or "error" in msg:
                        return msg
            return {}
        return r.json()

    def _sign_x402(self, payment_required_header: str) -> str:
        """Sign a USDC payment on Base L2 and return the X-PAYMENT header value."""
        try:
            from x402 import x402ClientSync
            from x402.http import decode_payment_required_header, safe_base64_encode
            from x402.mechanisms.evm.exact import ExactEvmClientScheme
            from eth_account import Account
        except ImportError:
            raise ImportError(
                "x402 autonomous payment requires: pip install 'x402[evm]' eth-account"
            )
        account = Account.from_key(self.x402_private_key)
        scheme = ExactEvmClientScheme(account)
        client = x402ClientSync()
        client.register("eip155:8453", scheme)
        payment_required = decode_payment_required_header(payment_required_header)
        payload = client.create_payment_payload(payment_required)
        return safe_base64_encode(payload.model_dump_json(by_alias=True, exclude_none=True))

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

        try:
            with httpx.Client(timeout=30) as client:
                self._ensure_session(client)
                r = client.post(
                    f"{self.endpoint}/mcp",
                    headers=self._build_headers(),
                    json=self._mcp_body(query),
                )

                if r.status_code in (400, 404):
                    # Session expired or server restarted — re-handshake once.
                    self._session_id = ""
                    self._ensure_session(client)
                    r = client.post(
                        f"{self.endpoint}/mcp",
                        headers=self._build_headers(),
                        json=self._mcp_body(query),
                    )

                if r.status_code == 402 and self.x402_private_key:
                    payment_required_header = r.headers.get("PAYMENT-REQUIRED", "")
                    if payment_required_header:
                        headers = self._build_headers()
                        headers["X-PAYMENT"] = self._sign_x402(payment_required_header)
                        r = client.post(
                            f"{self.endpoint}/mcp",
                            headers=headers,
                            json=self._mcp_body(query),
                        )

                if r.status_code == 402:
                    data = r.json()
                    action = data.get("action_for_agent", "")
                    raise RuntimeError(action or data.get("error", "Rate limit. Upgrade at https://graphifymd.com/pricing"))

                r.raise_for_status()
                msg = self._parse_rpc(r)
                if "error" in msg:
                    raise RuntimeError(msg["error"].get("message", "MCP error"))
                content = msg.get("result", {}).get("content", [{}])[0].get("text", "")
        except RuntimeError:
            raise
        except Exception as e:
            return [Document(page_content=f"CKG query failed: {e}", metadata={"domain": self.domain})]

        return [
            Document(
                page_content=content,
                metadata={
                    "domain": self.domain,
                    "concept": query,
                    "source": self.endpoint,
                    "payment_rail": "x402" if self.x402_private_key else (
                        "lightning" if self.lightning_invoice_id else (
                            "license" if self.license_key else "free"
                        )
                    ),
                },
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
