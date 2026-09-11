import numpy as np

from adaptive_search_rag.retrieval.rerank import (
    CrossEncoderReranker,
    LLMReranker,
    NoOpReranker,
)

DOCS = ["文档A", "文档B", "文档C"]


class _FakeCrossEncoder:
    """假交叉编码器：故意返回 numpy.ndarray（真实 predict 就是 ndarray），
    这样才测得到 float() 转换那条路径"""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return np.array(self.scores, dtype=np.float32)


def test_cross_encoder_rerank_sorts_by_score():
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])      # B 最高分
    reranker = CrossEncoderReranker(model=fake)

    result = reranker.rerank("任意问题", DOCS, top_k=2)

    assert result == ["文档B", "文档C"]
    # 断言确实把 [query, doc] 成对送进去了
    assert fake.calls[0] == [["任意问题", "文档A"], ["任意问题", "文档B"], ["任意问题", "文档C"]]


def test_cross_encoder_rerank_top_k_exceeds_docs():
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=fake)
    assert len(reranker.rerank("q", DOCS, top_k=10)) == 3


def test_cross_encoder_empty_docs():
    fake = _FakeCrossEncoder([])
    reranker = CrossEncoderReranker(model=fake)
    assert reranker.rerank("q", [], top_k=3) == []
    # 空候选池不应该触发加载
    assert fake.calls == []


def test_cross_encoder_returns_original_docs():
    fake = _FakeCrossEncoder([0.3, 0.2, 0.1])
    reranker = CrossEncoderReranker(model=fake)
    for doc in reranker.rerank("q", DOCS, top_k=3):
        assert doc in DOCS


class _FakeLLM:
    def __init__(self, scores):
        self.scores = scores
        self.i = 0

    def invoke(self, messages):
        score = self.scores[self.i]
        self.i += 1
        if score == "boom":
            raise RuntimeError("模拟 API 崩溃")
        return {"score": score}


def test_llm_reranker_sorts_by_score():
    reranker = LLMReranker(llm=_FakeLLM([0.2, 0.8, 0.5]))
    assert reranker.rerank("q", DOCS, top_k=2) == ["文档B", "文档C"]


def test_llm_reranker_survives_single_call_failure():
    """单条打分崩溃时记 0 分，整体不崩"""
    reranker = LLMReranker(llm=_FakeLLM([0.9, "boom", 0.5]))
    result = reranker.rerank("q", DOCS, top_k=3)
    assert result == ["文档A", "文档C", "文档B"]   # B 记 0 分排最后


def test_noop_reranker_keeps_order():
    assert NoOpReranker().rerank("q", DOCS, top_k=2) == ["文档A", "文档B"]
