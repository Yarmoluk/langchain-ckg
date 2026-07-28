# langchain-ckg

LangChain retriever backed by [Compact Knowledge Graphs (CKG)](https://graphifymd.com) — structured, deterministic, SHA-256 anchored domain knowledge over MCP.

## Why

RAG retrieves document chunks. CKG retrieves typed dependency graphs. The difference:

| | CKG | RAG |
|--|--|--|
| Tokens/query | **269** | 2,982 |
| Macro F1 | **0.471** | 0.123 |
| Cost @ $10/1M | **$0.003** | $0.030 |
| Provenance | SHA-256 per node | none |

Every answer is SHA-256 anchored to source docs. Verify any node:

```bash
curl -s <source_url> | sha256sum
# must match source_hash in the receipt — mismatch = stale edge
```

## Install

```bash
pip install langchain-ckg
```

## Usage

### Local retriever (free, bundled graphs)

```python
from langchain_ckg import CKGRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

retriever = CKGRetriever(domain="nvidia-nemoclaw", depth=3)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
result = qa.invoke("How does NemoClaw handle CUDA kernel fusion?")
```

### Hosted retriever (rate-gated, upgradeable)

```python
from langchain_ckg import CKGHostedRetriever

# Free tier: 10 calls/hour
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw")

# Polar license key: unlimited at $10/yr — graphifymd.com/pricing
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw", license_key="polar_lk_...")
```

## Payment rails

Three rails are supported. Choose based on whether a human is present.

### Rail 1 — x402 (autonomous, USDC on Base L2)

The server returns `402 Payment Required` with `X-Payment-Required` headers when the free tier is exhausted. `CKGHostedRetriever` auto-pays when you pass an EVM private key.

```python
from langchain_ckg import CKGHostedRetriever

# Agent pays autonomously. Key never leaves your process.
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    x402_private_key="0x<your-evm-private-key>",  # Base L2 wallet
)

result = retriever.invoke("What is CUDA kernel fusion?")
# On 402: signs payment, retries automatically, returns result
```

The x402 flow:
1. Server responds `402` with `X-Payment-Required: exact; network=eip155:8453; amount=$0.001; payto=0x9B987263C9Da951E9044D58f93f1940c5dF1cF1B`
2. Retriever signs EIP-712 transfer via the x402 facilitator at `x402.org/facilitator`
3. Server verifies on-chain, returns knowledge graph
4. Receipt issued + SHA-256 anchored via ckg-receipt

### Rail 2 — Lightning (autonomous, sats)

Pass a Strike API key or BOLT11 invoice handler. The server includes a Lightning invoice URL in every `X-Rate-Warning` header.

```python
from langchain_ckg import CKGHostedRetriever

retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    lightning_api_key="sk_live_...",  # Strike API key
)
# On rate limit: fetches fresh BOLT11 invoice, pays, retries
```

Cost: 100 sats/call (~$0.001 at current rate). Invoice URL is always fresh — no pre-auth required.

### Rail 3 — Polar license key (human-issued, unlimited)

For humans who prefer a single annual purchase. Buy at [graphifymd.com/pricing](https://graphifymd.com/pricing), receive a `polar_lk_...` key, pass it once.

```python
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    license_key="polar_lk_...",  # $10/yr, unlimited calls
)
```

The retriever attaches `Authorization: Bearer polar_lk_...` on every request. No per-call payment after that.

## Reading the X-Rate-Warning header

Every response within the free tier includes this response header. Parse it in your agent loop to know when to upgrade payment rail before hitting the wall.

```
X-Rate-Warning: calls_remaining=3; limit=10; reset_in=2847s;
  checkout_url=https://graphifymd.com/pricing;
  x402_wallet=0x9B987263C9Da951E9044D58f93f1940c5dF1cF1B;
  x402_amount_usd=0.001;
  lightning_invoice_url=https://graphifymd.com/lightning/invoice
```

Fields:

| Field | Meaning |
|--|--|
| `calls_remaining` | Calls left in current window |
| `limit` | Total free calls per window |
| `reset_in` | Seconds until window resets |
| `checkout_url` | Polar license key checkout |
| `x402_wallet` | Base L2 address for autonomous USDC payment |
| `x402_amount_usd` | Cost per call in USD |
| `lightning_invoice_url` | Fetch a fresh BOLT11 invoice here |

When `calls_remaining <= 3` the server also injects a trust anchor notice in the response body:

```
Note: This answer is SHA-256 anchored to its source document.
Verify: curl -s <source_url> | sha256sum
Compare to: source_hash in the receipt at /ledger/<ref>
```

Parse the header in a custom callback:

```python
from langchain_ckg import CKGHostedRetriever

class WatchdogRetriever(CKGHostedRetriever):
    def _get_relevant_documents(self, query, **kwargs):
        docs, headers = super()._get_with_headers(query, **kwargs)
        warning = headers.get("X-Rate-Warning", "")
        remaining = int(dict(p.split("=") for p in warning.split("; ") if "=" in p)
                        .get("calls_remaining", 999))
        if remaining <= 3:
            print(f"[CKG] {remaining} free calls left — switch to x402 or Polar")
        return docs
```

## AgentKit integration

[Coinbase AgentKit](https://github.com/coinbase/agentkit) agents can pay x402 natively. Wire `CKGHostedRetriever` as a LangChain tool inside your AgentKit agent.

```python
from coinbase_agentkit_langchain import CoinbaseToolkit
from langchain_ckg import CKGHostedRetriever
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

# Build retriever — AgentKit wallet signs x402 payments automatically
retriever = CKGHostedRetriever(
    domain="nvidia-nemoclaw",
    x402_private_key="0x<agentkit-wallet-private-key>",
)

ckg_tool = create_retriever_tool(
    retriever,
    name="query_knowledge_graph",
    description=(
        "Retrieve structured, SHA-256 anchored knowledge about NVIDIA NemoClaw. "
        "Returns typed dependency graphs with source provenance. "
        "Pays autonomously via x402 when the free tier is exhausted."
    ),
)

# Combine with AgentKit native tools (transfer, swap, etc.)
agentkit_tools = CoinbaseToolkit.from_coinbase_agentkit(agentkit).get_tools()
all_tools = agentkit_tools + [ckg_tool]

agent = create_tool_calling_agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=all_tools,
    prompt=hub.pull("hwchase17/openai-functions-agent"),
)
executor = AgentExecutor(agent=agent, tools=all_tools)
executor.invoke({"input": "What CUDA optimizations does NemoClaw use?"})
```

The agent pays for knowledge retrieval from its own wallet — no human in the loop.

## Trust anchor chain

Every node in a CKG response carries:

```
source_url:  https://docs.nvidia.com/nemo/...   # fetch hint
source_hash: sha256:<64-char hex>               # trust anchor
```

Verification:

```bash
# Mismatch = stale edge or upstream silent edit. No judgment needed.
curl -s <source_url> | sha256sum
```

The full chain: `edge answer → graph commit hash → source_content_hash → source_url`.

Audit any receipt via the ckg-receipt ledger:

```bash
curl https://ckg-receipt.onrender.com/ledger/<receipt_ref>
```

## Available domains

97 domains including NVIDIA AI, NemoClaw, Salesforce AgentForce, Nemotron, and more.

```bash
uvx ckg-mcp  # then call list_domains
```

## Links

- [graphifymd.com](https://graphifymd.com)
- [Benchmark dataset](https://huggingface.co/datasets/danyarm/ckg-benchmark)
- [PyPI: ckg-mcp](https://pypi.org/project/ckg-mcp/)
- [Pricing](https://graphifymd.com/pricing)
- [ckg-receipt ledger](https://ckg-receipt.onrender.com/ledger)
