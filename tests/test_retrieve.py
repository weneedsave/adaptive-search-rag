from adaptive_search_rag.nodes.retrieve import retrieve_node

def _state(question: str):
    """构造RAGState格式的字典，模拟graph传进来的state"""
    # RAGState 运行时就是 dict，四个键都得有
    return {"question": question, "route": "", "contexts": [], "answer": ""}
def test_retrieve_returns_contexts():
    # fake_retriever：伪造检索器，不碰真实Chroma，直接返回写死的chunk
    def fake_retriever(question, top_k=3):
        return ["片段A", "片段B", "片段C"]

    # 调用节点；传入state字典，注入fake_retriever，不走真实向量库
    result = retrieve_node(_state("什么是RAG"), top_k=3, retriever=fake_retriever)

    # 断言：节点输出应当把contexts填好
    assert result == {"contexts": ["片段A", "片段B", "片段C"]}
    assert isinstance(result["contexts"], list)
def test_retrieve_empty_library():
    def empty_retriever(question, top_k=3):
        return []                 # 模拟空库 / 没有匹配到任何chunk

    result = retrieve_node(_state("无关问题"), retriever=empty_retriever)
    assert result["contexts"] == []

