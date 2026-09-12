from adaptive_search_rag.nodes.search import search_node


def _state(question: str):
    return {"question": question, "contexts": [], "answer": ""}


def test_search_returns_contexts():
    def fake_search(query, top_k=3):
        return ["网络结果1", "网络结果2"]

    result = search_node(_state("今天天气"), top_k=3, search_fn=fake_search)
    assert result == {"contexts": ["网络结果1", "网络结果2"]}


def test_search_failure_degrades():
    def broken_search(query, top_k=3):
        raise RuntimeError("博查挂了")   # 模拟网络/API 失败
    # 关键断言：不崩，降级返回空 contexts
    result = search_node(_state("今天天气"), search_fn=broken_search)
    assert result["contexts"] == []
