<div align="center">

# langchain-ckg

**11 agent-stack knowledge graphs, bundled in the wheel. Offline, typed, SHA-256 anchored.**

[![PyPI](https://img.shields.io/pypi/v/langchain-ckg?color=0f6e56&label=PyPI)](https://pypi.org/project/langchain-ckg/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-ckg?color=0f6e56)](https://pypi.org/project/langchain-ckg/)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f6e56)](LICENSE)
[![Benchmark](https://img.shields.io/badge/F1-0.471%20vs%200.123-0f6e56)](https://huggingface.co/datasets/danyarm/ckg-benchmark)

</div>

---

Your coding assistant writes LangChain code from 2024. LangChain v1 renamed `create_react_agent` → `create_agent`, moved it to `langchain.agents`, renamed `prompt` → `system_prompt`, and exiled legacy chains to `langchain-classic` — and every model trained before late 2025 emits the old API by default, while vector indexes of old tutorials confirm it.

This package ships the antidote **inside the wheel**: pre-built knowledge graphs of the agent stack — LangGraph (including the v1 migration as typed `REPLACES` edges), MCP, agent memory, the major agent platforms — where every node carries the SHA-256 of the docs page it was extracted from. "Is this current?" becomes a mechanical check, not a hope.

```
RAG                                    CKG
────────────────────────────────────   ─────────────────────────────────────
Retrieve document chunks               Traverse typed dependency graph
Probabilistic similarity match         Deterministic BFS from matched concept
No provenance                          SHA-256 per node — verify any claim
2,982 tokens/query                     269 tokens/query  (11× cheaper)
F1 = 0.123                             F1 = 0.471  (4× better)
```

[Benchmark paper](https://github.com/Yarmoluk/ckg-benchmark/blob/main/paper/main.pdf) · [Dataset](https://huggingface.co/datasets/danyarm/ckg-benchmark) · patent-pending methodology

---

## Installation

```bash
pip install langchain-ckg
```

Depends only on `langchain-core` and `httpx`. The 11 graphs add ~120KB to the wheel.

## Instantiation

```python
from langchain_ckg import CKGRetriever, available_domains

retriever = CKGRetriever()          # bundled langgraph domain, offline
print(available_domains())          # all 11 bundled domains
```

## Usage

```python
docs = retriever.invoke("checkpointer")
print(docs[0].page_content)
# # Checkpointer (langgraph)
# ## Prerequisites
#   - MemorySaver
#     - StateGraph
#       - LangGraph Framework
# ## Builds toward
#   - thread_id
# Source: https://docs.langchain.com/oss/python/langgraph/persistence
# Source hash: sha256:114c96d9...

print(docs[0].metadata["source_url"])   # provenance in metadata too
```

Verify any answer against the live docs — no trust required:

```bash
curl -s https://docs.langchain.com/oss/python/langgraph/persistence | shasum -a 256
# equals the stored source_hash, or the edge is stale — deterministic either way
```

## Use within an agent

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ckg import CKGRetriever

retriever = CKGRetriever()

@tool
def langgraph_map(query: str) -> str:
    """Look up current LangGraph concepts, their prerequisites, and source docs."""
    return "\n\n".join(d.page_content for d in retriever.invoke(query))

agent = create_agent(model="anthropic:claude-opus-5", tools=[langgraph_map])
```

One-liner alternative: `from langchain_core.tools import create_retriever_tool` (works on langchain 0.3.x and 1.x).

## Bundled domains

| Domain | What it maps |
|---|---|
| `langgraph` *(default)* | StateGraph → checkpointers → streaming → multi-agent, **plus the v1 migration as `REPLACES` edges** |
| `mcp-protocol` | Model Context Protocol — hosts, servers, transports, tools |
| `agent-memory` | memory design patterns: episodic, semantic, working, cross-session |
| `agent-loop-patterns` | supervisor, worker, handoff, reflection loops |
| `aws-bedrock-agentcore` | AWS's agent runtime — gateways, identity, memory, tools |
| `google-gemini-agent-platform` | Vertex Agent Engine / Gemini agent stack |
| `a2a-protocol` | agent-to-agent protocol — cards, tasks, artifacts |
| `microsoft-ai-agent-stack` | Agent Framework (the Semantic Kernel + AutoGen successor) |
| `crewai` | crews, tasks, processes, flows |
| `llamaindex` | agents, workflows, query engines |
| `palantir-foundry` | ontology-first platform — Object/Link/Action Types |

Examples: [`examples/stale_api_correction.py`](examples/stale_api_correction.py) · [`examples/agent_memory_map.py`](examples/agent_memory_map.py) — both run offline with no API key.

## Hosted retriever — the full 100+ domain library

`CKGHostedRetriever` queries a hosted CKG MCP endpoint (NVIDIA stack, finance, healthcare, compliance domains). 48-hour free window, then `402` with upgrade options.

```python
from langchain_ckg import CKGHostedRetriever

retriever = CKGHostedRetriever(domain="nvidia-nemo")            # free 48h
retriever = CKGHostedRetriever(domain="nvidia-nemo",
                               license_key="CKGAP-...")         # graphifymd.com/pricing
```

<details>
<summary>Autonomous payment rails (x402 · Lightning · license key)</summary>

```python
# x402 — agent pays itself in USDC on Base L2 when it hits a 402
retriever = CKGHostedRetriever(
    domain="nvidia-nemo",
    x402_private_key="0x<evm-private-key>",   # pip install 'langchain-ckg[x402]'
)

# Lightning — pre-paid invoice, ~$0.001/call
retriever = CKGHostedRetriever(
    domain="nvidia-nemo",
    lightning_invoice_id="<invoice-id>",      # GET {endpoint}/lightning/invoice
)
```

| Tier | Price | Calls |
|------|-------|-------|
| Starter | $1 | 100 |
| Bundle | $4 | 500 |
| Dev | $29/mo | Unlimited |

Metered billing: `PolarUsageCallback(api_key=..., external_customer_id=...)` fires a Polar meter event per retrieval — pass via `retriever.invoke(query, config={"callbacks": [cb]})`.

</details>

## Trust anchor chain

Every node carries a SHA-256 hash of its source page bytes at extraction time:

```
source_url:  https://docs.langchain.com/oss/python/langgraph/...   ← fetch hint
source_hash: sha256:<64-char hex>                                  ← trust anchor
```

Full audit chain: `edge answer → graph commit → source_hash → source_url`. A hash mismatch means the upstream docs changed — the graph tells you it's stale instead of quietly guessing. Four bundled domains (`agent-memory`, `agent-loop-patterns`, `crewai`, `llamaindex`) currently carry extraction-internal references (`metadata.provenance = "extraction-internal"`) pending re-anchor to public URLs; the rest verify with `curl` today.

## API reference

`CKGRetriever(domain="langgraph", depth=3, k=5)` — bundled/local, sync `.invoke()` (async via default `ainvoke` delegation).
`CKGHostedRetriever(endpoint=..., domain=..., license_key=..., x402_private_key=..., lightning_invoice_id=..., depth=3, k=5)` — hosted MCP.
`available_domains() -> list[str]` — bundled domain names.

## Links

[graphifymd.com](https://graphifymd.com) · [Pricing](https://graphifymd.com/pricing) · [PyPI](https://pypi.org/project/langchain-ckg/) · [Benchmark](https://github.com/Yarmoluk/ckg-benchmark/blob/main/paper/main.pdf) · [Dataset](https://huggingface.co/datasets/danyarm/ckg-benchmark)
