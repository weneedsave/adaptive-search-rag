# evaluation/tests/test_stubs.py
from functools import partial
from adaptive_search_rag.nodes.search import search_node
from evaluation.stubs import (
    STUB_MARKER,
    build_eval_graph,
    contains_stub,
    stub_search_fn,
)


def test_stub_returns_one_marked_string():
    out = stub_search_fn("北京天气")
    assert len(out) == 1
    assert STUB_MARKER in out[0]


def test_stub_records_query():
    # search_node 用的是 state["question"] 而非 retrieval_query，
    # 桩把 query 记下来，这个设计决策就被真实验证了
    assert "北京天气" in stub_search_fn("北京天气")[0]


def test_contains_stub():
    assert contains_stub(["普通片段", STUB_MARKER + "（query=x）"]) is True
    assert contains_stub(["普通片段"]) is False
    assert contains_stub([]) is False


def test_real_search_node_splicing_is_used():
    """核心：跑的是真 search_node，拼接逻辑是真的。"""
    node = partial(search_node, search_fn=stub_search_fn)
    out = node({"question": "北京天气", "contexts": ["本地片段A"]})
    # existing + new，不是 new 覆盖 existing
    assert out["contexts"][0] == "本地片段A"
    assert STUB_MARKER in out["contexts"][1]


def test_real_search_node_failure_keeps_existing():
    """search_fn 抛异常时，真节点返回 existing（不是 []）。"""
    def boom(query, top_k=3):
        raise RuntimeError("桩挂了")
    node = partial(search_node, search_fn=boom)
    out = node({"question": "q", "contexts": ["本地片段A"]})
    assert out["contexts"] == ["本地片段A"]


def test_build_eval_graph_compiles():
    app = build_eval_graph()
    assert app is not None
