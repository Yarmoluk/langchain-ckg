"""
CKGHostedRetriever + Coinbase AgentKit — autonomous x402 payment example.

Shows two approaches to autonomous payment:

1. CKGHostedRetriever with x402_private_key — auto-signs USDC on Base L2
   whenever the hosted CKG returns a 402. No human required.

2. Coinbase AgentKit agent with CKGHostedRetriever — AgentKit's built-in
   x402 support handles the payment loop natively.

Install:
    pip install langchain-ckg[x402] coinbase-agentkit langchain-openai

Run:
    # Approach 1 — standalone retriever with x402 key
    PRIVATE_KEY=0x... python agentkit_ckg_example.py

    # Approach 2 — full AgentKit agent (needs CDP_API_KEY_NAME + CDP_API_KEY_PRIVATE_KEY)
    CDP_API_KEY_NAME=... CDP_API_KEY_PRIVATE_KEY=... OPENAI_API_KEY=... python agentkit_ckg_example.py --agent
"""

import os
import sys

# ---------------------------------------------------------------------------
# Approach 1: CKGHostedRetriever with x402 auto-pay
# ---------------------------------------------------------------------------

def demo_retriever_x402():
    """Retriever auto-pays $0.001 USDC on Base L2 when rate-limited."""
    from langchain_ckg import CKGHostedRetriever

    private_key = os.environ.get("PRIVATE_KEY", "")
    if not private_key:
        print("Set PRIVATE_KEY=0x... to enable x402 auto-pay. Running without payment key (will hit rate limit).")

    retriever = CKGHostedRetriever(
        endpoint="https://ckg-nvidia-ai.onrender.com",
        domain="nvidia-nemo",
        x402_private_key=private_key,
        depth=3,
    )

    print("Querying: 'CUDA kernel fusion'...")
    docs = retriever.invoke("CUDA kernel fusion")
    for doc in docs:
        print(f"\n[payment_rail={doc.metadata.get('payment_rail', 'unknown')}]")
        print(doc.page_content[:500])


# ---------------------------------------------------------------------------
# Approach 2: AgentKit agent with CKGHostedRetriever tool
# ---------------------------------------------------------------------------

def demo_agentkit_agent():
    """Full AgentKit agent that uses CKG as a knowledge tool with x402 payments."""
    try:
        from coinbase_agentkit import CoinbaseAgentKit, CoinbaseAgentKitConfig
        from coinbase_agentkit_langchain import CoinbaseAgentKitToolkit
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.tools.retriever import create_retriever_tool
    except ImportError:
        print("Install: pip install coinbase-agentkit coinbase-agentkit-langchain langchain-openai")
        return

    from langchain_ckg import CKGHostedRetriever

    agentkit = CoinbaseAgentKit(
        CoinbaseAgentKitConfig(
            cdp_api_key_name=os.environ["CDP_API_KEY_NAME"],
            cdp_api_key_private_key=os.environ["CDP_API_KEY_PRIVATE_KEY"],
            network_id="base-mainnet",
        )
    )
    agentkit_tools = CoinbaseAgentKitToolkit(agentkit=agentkit).get_tools()

    wallet_address = agentkit.wallet.default_address.address_id
    print(f"Agent wallet: {wallet_address}")

    retriever = CKGHostedRetriever(
        endpoint="https://ckg-nvidia-ai.onrender.com",
        domain="nvidia-nemo",
        x402_private_key=agentkit.wallet.default_address.export(),
        depth=3,
    )

    ckg_tool = create_retriever_tool(
        retriever,
        name="query_ckg",
        description=(
            "Query the NVIDIA NemoClaw knowledge graph. Returns prerequisite concepts and "
            "dependency chains for any NVIDIA NemoClaw / CUDA / inference concept. "
            "Pays autonomously via x402 ($0.001 USDC on Base L2) if rate-limited."
        ),
    )

    all_tools = agentkit_tools + [ckg_tool]

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI agent with access to an NVIDIA knowledge graph. "
            "When queried about NVIDIA NemoClaw, CUDA, or inference optimization, "
            "use the query_ckg tool to get accurate, deterministic answers. "
            "You can also use your wallet tools to pay for services autonomously."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

    result = executor.invoke({"input": "What are the prerequisites for CUDA kernel fusion in NemoClaw?"})
    print("\nAgent response:", result["output"])


if __name__ == "__main__":
    if "--agent" in sys.argv:
        demo_agentkit_agent()
    else:
        demo_retriever_x402()
