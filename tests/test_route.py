from adaptive_search_rag.nodes.route import route_node


def _state(question: str):
    return {"question": question, "route": "", "contexts": [], "answer": ""}


class _FakeLLM:
    """假的『结构化输出』LLM：invoke(messages) 返回固定 dict"""
    def __init__(self, route: str):
        self._route = route

    def invoke(self, messages):
        return {"route": self._route}


def test_route_to_retrieve():
    result = route_node(_state("什么是RAG"), llm=_FakeLLM("retrieve"))
    assert result == {"route": "retrieve"}


def test_route_to_search():
    result = route_node(_state("今天股市行情"), llm=_FakeLLM("search"))
    assert result == {"route": "search"}
