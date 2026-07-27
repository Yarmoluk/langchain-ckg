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

# Free: 50 calls/day
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw")

# Unlimited: $10/yr at graphifymd.com/pricing
retriever = CKGHostedRetriever(domain="nvidia-nemoclaw", license_key="your-key")
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
