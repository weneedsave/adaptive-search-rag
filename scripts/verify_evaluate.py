# scripts/verify_evaluate.py
"""验证 evaluate_node 的三档打标是否与 should_act 的映射一致。

期望（见 graph.should_act）：
  correct   -> generate
  ambiguous -> search   （部分相关，补充联网）
  incorrect -> rewrite  （完全无关，先改写重试本地）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.evaluate import evaluate_node

QUESTION = "什么是RAG？"

# 三段资料，相关性递减
CTX_CORRECT = """RAG（检索增强生成）是一种把外部知识库检索与大模型生成相结合的技术。
它在生成答案前，先从向量数据库中检索相关文档片段，再把这些片段作为上下文交给大模型，
从而缓解大模型的幻觉问题，并让答案可以溯源。"""

CTX_AMBIGUOUS = """向量数据库是 RAG 系统的核心组件之一，负责存储文档的嵌入向量并做相似度检索。
RAG 系统通常还包含一个生成模块，把检索到的片段拼进提示词交给大模型。"""

CTX_INCORRECT = """今天北京天气晴朗，气温 18 到 26 摄氏度，空气质量良好，适合户外活动。
明天有一股冷空气南下，气温将下降 5 到 7 度，请注意添衣。"""

CASES = [
    ("资料充足", CTX_CORRECT, "correct"),
    ("部分相关", CTX_AMBIGUOUS, "ambiguous"),
    ("完全无关", CTX_INCORRECT, "incorrect"),
    ("检索为空", "", "(空 contexts 短路)"),
]


def _state(contexts):
    ctx = [contexts] if contexts else []
    return {"question": QUESTION, "retrieval_query": "", "contexts": ctx,
            "grade": "", "retry_count": 0, "answer": ""}


print(f"问题：{QUESTION}\n")
passed = 0
for label, ctx, expected in CASES:
    result = evaluate_node(_state(ctx))
    got = result["grade"]
    cleared = "contexts已清空" if result.get("contexts") == [] else "contexts保留"
    if expected == "(空 contexts 短路)":
        mark = "✓" if got == "incorrect" else "✗"
    else:
        mark = "✓" if got == expected else "✗"
    passed += mark == "✓"
    print(f"  {mark} {label:8s} 期望={expected:10s} 实际={got:10s} [{cleared}]")

print(f"\n{passed}/{len(CASES)} 符合预期")
