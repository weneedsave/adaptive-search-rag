# scripts/verify_rewrite.py
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.rewrite import rewrite_query_node

# 疑问词：中文 BM25 下这些 token 会带偏排序（见踩坑记录 #3：「什么」IDF=2.27）
INTERROGATIVES = ["什么", "怎么", "为什么", "如何", "吗", "哪些", "哪个", "是否"]

cases = [
    "RAG它解决了大模型什么问题？",          # prompt 里的示例原句
    "什么是LangGraph的checkpoint？",
    "我该怎么用BM25做中文检索？",
    "rerank为什么要放在最后？",
    "SqliteSaver是怎么保存会话记忆的？",
]


def _state(q):
    return {"question": q, "retrieval_query": "", "contexts": [],
            "grade": "", "retry_count": 0, "answer": ""}


for q in cases:
    new_q = rewrite_query_node(_state(q))["retrieval_query"]
    hits = [w for w in INTERROGATIVES if w in new_q]
    flag = f"   ⚠ 残留疑问词 {hits}" if hits else "   ✓ 无疑问词"
    print(f"原问题：{q}")
    print(f"改写后：{new_q}{flag}")
    print()
