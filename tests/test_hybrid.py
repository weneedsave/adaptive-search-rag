from adaptive_search_rag.retrieval.hybrid import (
    BM25_K,
    CANDIDATE_K,
    VECTOR_K,
    build_hybrid_retriever,
)


class _RecordingRetriever:
    """记录被调用参数的假粗筛器，返回固定排名"""
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, question, top_k):
        self.calls.append((question, top_k))
        return self.result[:top_k]


class _RecordingReranker:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def rerank(self, question, docs, top_k):
        self.calls.append((question, list(docs), top_k))
        if self.result is not None:
            return self.result
        return docs[:top_k]


def test_hybrid_orchestration_order_and_kwargs():
    vector = _RecordingRetriever(["V1", "V2", "V3", "V4", "V5"])
    bm25 = _RecordingRetriever(["B1", "B2"])
    reranker = _RecordingReranker()
    retriever = build_hybrid_retriever(vector, bm25, reranker)
    result = retriever("问题", top_k=2)

    # 1. 两路粗筛各自用了正确的 K
    assert vector.calls == [("问题", VECTOR_K)]
    assert bm25.calls == [("问题", BM25_K)]

    # 2. 送进 rerank 的是 RRF 融合后截断的候选池
    q, docs, k = reranker.calls[0]
    assert q == "问题"
    assert len(docs) <= CANDIDATE_K
    assert set(docs) == {"V1", "V2", "V3", "V4", "V5", "B1", "B2"}

    # 3. 最终条数由 top_k 决定
    assert k == 2
    assert result == docs[:2]


def test_hybrid_uses_reranker_output_verbatim():
    """hybrid 不加工 reranker 的返回值"""
    vector = _RecordingRetriever(["V1", "V2"])
    bm25 = _RecordingRetriever(["B1"])
    reranker = _RecordingReranker(result=["重排后的唯一结果"])
    result = build_hybrid_retriever(vector, bm25, reranker)("q", top_k=3)
    assert result == ["重排后的唯一结果"]


def test_hybrid_both_sides_empty():
    vector = _RecordingRetriever([])
    bm25 = _RecordingRetriever([])
    reranker = _RecordingReranker()
    result = build_hybrid_retriever(vector, bm25, reranker)("q", top_k=3)
    assert result == []
    assert reranker.calls[0][1] == []      # 送进 rerank 的是空候选池


def test_hybrid_default_top_k_is_three():
    vector = _RecordingRetriever([f"V{i}" for i in range(10)])
    bm25 = _RecordingRetriever([f"B{i}" for i in range(10)])
    reranker = _RecordingReranker()
    build_hybrid_retriever(vector, bm25, reranker)("q")
    assert reranker.calls[0][2] == 3
