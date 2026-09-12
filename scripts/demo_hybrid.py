"""真实对比演示：纯向量 vs BM25 vs 混合+精排

用法：.venv/Scripts/python.exe scripts/demo_hybrid.py
首次运行要加载两个模型，等十几秒属正常。

面试用法：挑一个「含精确术语」的问题，指出 BM25 把它提上来了，
而纯向量检索可能排在后面——这就是混合检索的价值。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from chromadb.utils import embedding_functions

from adaptive_search_rag.nodes.retrieve import build_retriever, build_vector_retriever
from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever
from adaptive_search_rag.retrieval.hybrid import BM25_K, VECTOR_K

# 挑问题：优先选含【专有名词/术语】的，这类问题最能拉开 BM25 和向量的差距
QUESTIONS = [
    "StateGraph 是什么",                       # 精确术语
    "checkpoint 和 store 有什么区别",          # 两个并列术语
    "什么是 RAG 的幻觉问题",                   # 语义型问题，两路应该接近
]


def _load_chunks() -> list[str]:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5", local_files_only=True
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="adaptive_knowledge", embedding_function=ef)
    return collection.get()["documents"]


def _brief(doc: str, width: int = 46) -> str:
    """把 chunk 压成一行摘要，方便并排看"""
    return doc.replace("\n", " ").strip()[:width]


def _show(label: str, docs: list[str]) -> None:
    print(f"  {label}")
    for i, d in enumerate(docs, 1):
        print(f"    {i}. {_brief(d)}")


def main() -> None:
    vector = build_vector_retriever()
    bm25 = build_bm25_retriever(_load_chunks())
    hybrid = build_retriever()          # 带 lru_cache，不会重复加载模型

    for q in QUESTIONS:
        print("=" * 78)
        print(f"问题：{q}")
        print("=" * 78)

        t0 = time.time()
        v = vector(q, VECTOR_K)
        t_vec = time.time() - t0
        _show(f"[纯向量 top-3]  ({t_vec*1000:.0f} ms 取10条)", v[:3])

        t0 = time.time()
        b = bm25(q, BM25_K)
        t_bm = time.time() - t0
        _show(f"[BM25  top-3]   ({t_bm*1000:.0f} ms 取10条)", b[:3])

        t0 = time.time()
        h = hybrid(q, 3)
        t_hy = time.time() - t0
        _show(f"[混合+精排 top-3] ({t_hy:.1f} s)", h)

        # 高亮差异：BM25 提到了向量 top-3 之外的内容
        v_top3, b_top3 = set(v[:3]), set(b[:3])
        only_bm25 = b_top3 - v_top3
        only_vec = v_top3 - b_top3
        if only_bm25:
            print(f"  >> BM25 独有（向量没召回）: {len(only_bm25)} 条")
        if only_vec:
            print(f"  >> 向量独有（BM25 没召回）: {len(only_vec)} 条")
        if not only_bm25 and not only_vec:
            print("  >> 两路 top-3 完全一致")
        print()


if __name__ == "__main__":
    main()
