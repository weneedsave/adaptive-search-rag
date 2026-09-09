# scripts/verify_web_search_fn.py
from adaptive_search_rag.tools.web_search import web_search

results = web_search("什么是RAG", 3)
print(f"返回 {len(results)} 条：")
for i, r in enumerate(results):
    print(f"\n--- 片段 {i} ---")
    print(r[:200])
