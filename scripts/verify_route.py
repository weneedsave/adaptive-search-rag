# scripts/verify_route.py
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.route import route_node

cases = [
    "什么是RAG",                          # 期望 retrieve
    "LangGraph 的节点和边是什么",          # 期望 retrieve
    "checkpoint 和 store 有什么区别",      # 期望 retrieve
    "今天北京天气怎么样",                  # 期望 search
    "2026年最新的新能源汽车销量",          # 期望 search
]

for q in cases:
    result = route_node({"question": q, "route": "", "contexts": [], "answer": ""})
    print(f"{q}  →  {result['route']}")
