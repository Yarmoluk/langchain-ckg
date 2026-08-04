"""Example 2 — "why does my agent forget everything?"

The most-asked LangGraph production question. The answer is a dependency
chain, not a document: thread-scoped persistence REQUIRES a Checkpointer;
PostgresSaver needs a thread_id in config; cross-session memory is a
DIFFERENT interface (BaseStore), not the checkpointer. Vector retrieval
answers this badly because similarity mixes in deprecated
ConversationBufferMemory tutorials; a typed graph returns the minimal
prerequisite chain in a few hundred tokens.

Runs fully offline — no API key, no network.
"""

from langchain_ckg import CKGRetriever

retriever = CKGRetriever(depth=3)  # bundled langgraph domain

for query in ("checkpointer", "BaseStore"):
    print(f"### query: {query}\n")
    for doc in retriever.invoke(query):
        print(doc.page_content)
        print("-" * 60)
    print()

# Related bundled domain: framework-agnostic memory design patterns
#   CKGRetriever(domain="agent-memory").invoke("episodic memory")
