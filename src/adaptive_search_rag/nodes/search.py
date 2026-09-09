from adaptive_search_rag.state import RAGState
from adaptive_search_rag.tools.web_search import web_search


def search_node(state: RAGState, top_k: int = 3, search_fn=None) -> dict:
    #state:传入状态包含`question, route, contexts, answer`
    if search_fn is None:
        search_fn = web_search
    question = state["question"]
    #   成功 → return {"contexts": search_fn(question, top_k)}
    #   失败 → 打印/记录错误，return {"contexts": []}  （spec §7：不让图崩）
    try:
        web_contexts = search_fn(question, top_k)
        return {"contexts": web_contexts}
    except Exception as e:
        print(f"联网搜索出错：{e}")
        return {"contexts": []}
