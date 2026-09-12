"""混合检索探活：真实加载两个模型，端到端验证 build_retriever。

用法：.venv/Scripts/python.exe scripts/smoke_hybrid.py
首次运行要加载 Embedding + CrossEncoder，等十几秒属正常。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.retrieve import build_retriever
from adaptive_search_rag.retrieval.hybrid import BM25_K, CANDIDATE_K, VECTOR_K
from adaptive_search_rag.retrieval.rerank import CrossEncoderReranker

QUESTION = "什么是 RAG 的幻觉问题"


def main() -> None:
    print(f"常量: VECTOR_K={VECTOR_K}  BM25_K={BM25_K}  CANDIDATE_K={CANDIDATE_K}")
    print()

    # ---- 冷启动：真实加载 Embedding + BM25 索引 + CrossEncoder ----
    t0 = time.time()
    retriever = build_retriever()
    print(f"[1] 冷启动 build_retriever 耗时: {time.time() - t0:.1f}s")

    # ---- 检查有没有偷偷降级 ----
    # build_retriever 已经组装完，reranker 藏在闭包里；
    # 用 closure 取出来看它到底是哪个类
    cell = retriever.__closure__
    reranker = None
    for c in cell:
        if isinstance(c.cell_contents, CrossEncoderReranker):
            reranker = c.cell_contents
            break
    if reranker is None:
        print("[!] 警告：未找到 CrossEncoderReranker —— 很可能已降级为 NoOpReranker！")
        print("    (上面应该打印过 [警告] 开头的降级日志，往上翻)")
    else:
        print(f"[2] reranker 实际类型: {type(reranker).__name__}  ✓ 精排生效")

    print()

    # ---- 真实检索 ----
    t0 = time.time()
    docs = retriever(QUESTION, top_k=3)
    print(f"[3] 检索「{QUESTION}」耗时: {time.time() - t0:.2f}s  返回 {len(docs)} 条")
    for i, d in enumerate(docs, 1):
        print(f"  --- top{i} --- {d[:100]}")

    print()

    # ---- 验证 lru_cache 生效 ----
    t0 = time.time()
    build_retriever()
    print(f"[4] 再次 build_retriever 耗时: {time.time() - t0:.4f}s  "
          f"（接近 0 说明 lru_cache 生效）")


if __name__ == "__main__":
    main()
