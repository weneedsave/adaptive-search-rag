# scripts/verify_generate.py
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.generate import generate_node
from adaptive_search_rag.nodes.retrieve import retrieve_node

question = "什么是RAG"

# 第一步：真实检索，拿到 contexts
ret = retrieve_node({"question": question, "route": "", "contexts": [], "answer": ""})
print(f"检索到 {len(ret['contexts'])} 条上下文\n")

# 第二步：真实生成，基于 contexts 生成答案
state = {"question": question, "route": "", "contexts": ret["contexts"], "answer": ""}
result = generate_node(state)

print("=" * 60)
print("最终答案：")
print(result["answer"][:500])
