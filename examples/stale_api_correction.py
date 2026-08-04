"""Example 1 — deterministic stale-API correction.

Every model trained before late 2025 writes LangGraph agents like this:

    from langgraph.prebuilt import create_react_agent      # moved in v1
    agent = create_react_agent(model, tools, prompt=...)   # renamed in v1

LangChain v1 renamed create_react_agent -> create_agent, moved it to
langchain.agents, renamed prompt -> system_prompt, and replaced hooks with
middleware. A vector index of old tutorials happily confirms the stale API.
The bundled graph declares the migration as typed edges instead — and every
node carries the SHA-256 of the migration page it was extracted from, so
"is this current?" is a mechanical check, not a hope.

Runs fully offline — no API key, no network.
"""

from langchain_ckg import CKGRetriever

retriever = CKGRetriever()  # bundled langgraph domain

for doc in retriever.invoke("create_agent"):
    print(doc.page_content)
    print("-" * 60)
    print("verify:", f"curl -s {doc.metadata['source_url']} | shasum -a 256")
    print("expect:", doc.metadata["source_hash"])
    print("=" * 60)

# To wire this into an agent as its "current API" reference (langchain >= 1.0):
#
#     from langchain.agents import create_agent
#     from langchain.tools import tool
#
#     @tool
#     def langgraph_api_map(query: str) -> str:
#         """Look up current LangGraph/LangChain v1 API concepts and what they replaced."""
#         return "\n\n".join(d.page_content for d in retriever.invoke(query))
#
#     agent = create_agent(model="anthropic:claude-opus-5", tools=[langgraph_api_map])
