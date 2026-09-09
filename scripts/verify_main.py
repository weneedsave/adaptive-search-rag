import sys
sys.stdout.reconfigure(encoding="utf-8")
from adaptive_search_rag.main import ask

# 1. 基本问答
print("=== 基本问答 ===")
print(ask("什么是RAG", thread_id="t1")[:200])

# 2. 同一个 thread_id 连续问，验证状态持久化
print("\n=== 同会话 t2 连续两问 ===")
print(ask("什么是RAG", thread_id="t2")[:100])
print(ask("它解决了什么问题", thread_id="t2")[:100])  # "它"指代上文 RAG

# 3. 不同 thread_id 隔离
print("\n=== 跨会话隔离 ===")
print(ask("它是什么", thread_id="t3")[:100])  # 新会话，没有上文
