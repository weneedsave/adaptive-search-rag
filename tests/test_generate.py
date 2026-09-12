from adaptive_search_rag.nodes.generate import generate_node


def _state(question, contexts):
    return {"question": question, "contexts": contexts, "answer": ""}


class _FakeLLM:
    """假的裸 LLM：记录收到的 messages，返回带 .content 的对象"""#模拟 LangGraph 传给节点的 state
    def __init__(self, answer="这是答案"):
        self._answer = answer
        self.messages = None          # 记录 invoke 收到的参数，供断言

    def invoke(self, messages):
        self.messages = messages
        return _FakeAIMessage(self._answer)


class _FakeAIMessage:
    def __init__(self, content):
        self.content = content


def test_generate_with_contexts():
    fake = _FakeLLM(answer="RAG 是检索增强生成")
    #有参考资料
    result = generate_node(_state("什么是RAG", ["片段1", "片段2"]), llm=fake)
    assert result == {"answer": "RAG 是检索增强生成"}
    # 关键：断言 prompt 里包含了 question 和 contexts
    all_text = "".join(m.content for m in fake.messages)
    assert "什么是RAG" in all_text          # 含问题
    assert "片段1" in all_text              # 含上下文1
    assert "片段2" in all_text              # 含上下文2


def test_generate_empty_contexts():
    # 空 contexts：不调 LLM，直接返回提示
    fake = _FakeLLM()
    result = generate_node(_state("什么是RAG", []), llm=fake)
    assert fake.messages is None            # 证明没调 LLM
    assert "根据现有资料无法回答" in result["answer"]
