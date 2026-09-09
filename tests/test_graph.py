from adaptive_search_rag.graph import build_graph

# ===== 四个假节点：记录被调用，返回写死数据 =====
def _fake_route(state):
    return {"route": state["question"]}   # 把 question 直接当 route 值，方便控制分支

def _fake_retrieve(state):
    return {"contexts": ["本地文档片段"]}

def _fake_search(state):
    return {"contexts": ["联网搜索结果"]}

def _fake_generate(state):
    return {"answer": f"回答：{state['contexts']}"}

def _build_fake_graph():
    return build_graph(nodes={
        "route": _fake_route,
        "retrieve": _fake_retrieve,
        "search": _fake_search,
        "generate": _fake_generate,
    })

def test_route_to_retrieve_branch():
    g = _build_fake_graph()
    state = {"question": "retrieve", "route": "", "contexts": [], "answer": ""}
    result = g.invoke(state)
    assert result["answer"] == "回答：['本地文档片段']"   # 走了 retrieve 分支

def test_route_to_search_branch():
    g = _build_fake_graph()
    state = {"question": "search", "route": "", "contexts": [], "answer": ""}
    result = g.invoke(state)
    assert result["answer"] == "回答：['联网搜索结果']"   # 走了 search 分支
from adaptive_search_rag.graph import path_fn

def test_path_fn_fallback_on_invalid_route():
    # LLM 偶尔输出非法值（如 "retrieval"），path_fn 应归入 retrieve
    assert path_fn({"route": "retrieval", "question": "", "contexts": [], "answer": ""}) == "retrieve"
    assert path_fn({"route": "", "question": "", "contexts": [], "answer": ""}) == "retrieve"
    assert path_fn({"route": "search", "question": "", "contexts": [], "answer": ""}) == "search"
