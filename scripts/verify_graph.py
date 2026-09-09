# scripts/verify_graph.py
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.graph import build_graph

g = build_graph()   # 真实图，不注入假节点

for q in ["什么是RAG", "今天北京天气怎么样"]:
    result = g.invoke({"question": q, "route": "", "contexts": [], "answer": ""})
    print(f"问题：{q}")
    print(f"路由：{result['route']}")
    print(f"答案：{result['answer'][:200]}")
    print("=" * 60)
