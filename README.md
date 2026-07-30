<div align="center">

# langchain-ckg

**LangChain retriever backed by [Compact Knowledge Graphs](https://graphifymd.com) — deterministic, SHA-256 anchored, 4× better than RAG**

[![PyPI](https://img.shields.io/pypi/v/langchain-ckg?color=0f6e56&label=PyPI)](https://pypi.org/project/langchain-ckg/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-ckg?color=0f6e56)](https://pypi.org/project/langchain-ckg/)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f6e56)](LICENSE)
[![Benchmark](https://img.shields.io/badge/F1-0.471%20vs%200.123-0f6e56)](https://huggingface.co/datasets/danyarm/ckg-benchmark)

</div>

---

## CKG vs RAG — the difference that matters

```
RAG                                    CKG
────────────────────────────────────   ─────────────────────────────────────
Retrieve document chunks               Traverse typed dependency graph
Probabilistic similarity match         Deterministic BFS from matched concept
No provenance                          SHA-256 per node — verify any claim
2,982 tokens/query                     269 tokens/query  (11× cheaper)
F1 = 0.123                             F1 = 0.471  (4× better)
```

| Metric | CKG | RAG | GraphRAG |
|--------|-----|-----|----------|
| Macro F1 | **0.471** | 0.123 | 0.120 |
| Tokens/query | **269** | 2,982 | 2,982 |
| Cost @ $10/1M | **$0.003** | $0.030 | $0.030 |
| Source provenance | SHA-256 per node | none | none |

[Benchmark paper](https://github.com/Yarmoluk/ckg-benchmark/blob/main/paper/main.pdf) · [Dataset](https://huggingface.co/datasets/danyarm/ckg-benchmark) · patent-pending methodology

---

## Install

```bash
pip install langchain-ckg
```

---

## Usage

### Local retriever — free, no network

Queries run entirely in-process against graphs bundled in `ckg-mcp`. No API key, no rate limit.

```python
from langchain_ckg import CKGRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

retriever = CKGRetriever(domain="nvidia-nemoclaw", depth=3)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
result = qa.invoke("How does NemoClaw handle CUDA kernel fusion?")
```

### Hosted retriever — 48-hour free trial, then $1 / 100 calls

```python
from langchain_ckg import CKGHostedRetriever

# Free for 48 hours from first call
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw")

# After trial: pass a license key
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    license_key="CKGAP-...",  # graphifymd.com/pricing
)
```

After 48 hours the server returns `402 Payment Required` with upgrade options. See [Payment rails](#payment-rails) for autonomous payment.

---

## Payment rails

### x402 — autonomous USDC on Base L2

Agent pays itself. No human in the loop.

```python
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    x402_private_key="0x<evm-private-key>",
)
# On 402: signs $0.010 USDC payment, retries, returns result
```

### License key — human-purchased, unlimited

[graphifymd.com/pricing](https://graphifymd.com/pricing) → receive `CKGAP-...` key → pass once:

```python
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw", license_key="CKGAP-...")
```

| Tier | Price | Calls |
|------|-------|-------|
| Starter | $1 | 100 |
| Bundle | $4 | 500 |
| Dev | $29/mo | Unlimited |

### Lightning — autonomous sats

```python
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    lightning_invoice_id="<invoice-id>",  # GET {endpoint}/lightning/invoice
)
# 100 sats/call (~$0.001)
```

---

## Metered billing with PolarUsageCallback

Track per-retrieval usage in Polar — fires a background thread, never blocks the chain:

```python
from langchain_ckg import CKGRetriever, PolarUsageCallback

cb = PolarUsageCallback(api_key="sk_...", external_customer_id="license_key_abc")
retriever = CKGRetriever(domain="nvidia-nemoclaw", depth=3)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever, callbacks=[cb])
```

---

## AgentKit integration

Coinbase AgentKit agents pay x402 natively. Wire `CKGHostedRetriever` as a LangChain tool:

```python
from coinbase_agentkit_langchain import CoinbaseToolkit
from langchain_ckg import CKGHostedRetriever
from langchain.tools.retriever import create_retriever_tool

retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    x402_private_key="0x<agentkit-wallet-key>",
)
ckg_tool = create_retriever_tool(
    retriever,
    name="query_knowledge_graph",
    description="Retrieve SHA-256 anchored knowledge about NVIDIA NemoClaw. Pays autonomously via x402.",
)
# Combine with AgentKit tools and run
```

---

## Trust anchor chain

Every node carries a SHA-256 hash of its source document bytes at extraction time:

```
source_url:  https://docs.nvidia.com/nemo/...   ← fetch hint
source_hash: sha256:<64-char hex>               ← trust anchor
```

Verify any claim:

```bash
curl -s <source_url> | sha256sum
# mismatch = stale edge or upstream silent edit — no judgment needed
```

Full audit chain: `edge answer → graph commit → source_hash → source_url`

---

## Available domains

**117 domains** — NVIDIA AI stack, NemoClaw, Salesforce AgentForce, finance (Basel III · SEC · IFRS), agent protocols (MCP · A2A · x402), infrastructure (Render · Stripe · PostHog · Cloudflare), and more.

```bash
uvx ckg-mcp  # → call list_domains
```

---

## Links

[graphifymd.com](https://graphifymd.com) · [Pricing](https://graphifymd.com/pricing) · [PyPI](https://pypi.org/project/langchain-ckg/) · [Benchmark](https://github.com/Yarmoluk/ckg-benchmark/blob/main/paper/main.pdf) · [Dataset](https://huggingface.co/datasets/danyarm/ckg-benchmark)
