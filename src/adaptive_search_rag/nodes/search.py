from adaptive_search_rag.state import RAGState
from adaptive_search_rag.tools.web_search import web_search


def search_node(state: RAGState, top_k: int = 3, search_fn=None) -> dict:
    """联网搜索节点：联网结果追加到已有上下文，API失败保留本地contexts"""
    if search_fn is None:
        search_fn = web_search

    question = state["question"]   # 联网用用户原始问题，不使用改写检索词
    existing = state.get("contexts", [])  # 取出之前本地检索得到的片段

    try:
        new_web_contexts = search_fn(question, top_k)
        # 原有本地片段 + 新的网络搜索结果，合并
        return {"contexts": existing + new_web_contexts}
    except Exception as e:
        print(f"联网搜索出错：{e}")
        # 网络挂了，直接返回已有的本地上下文，保住本地结果！不是空列表[]
        return {"contexts": existing}

#最终返回本地检索的结果和联网检索的结果

