from langgraph.graph import StateGraph, START, END
#state
from adaptive_search_rag.state import RAGState
#判断是否联网
from adaptive_search_rag.nodes.route import route_node
#本地搜索
from adaptive_search_rag.nodes.retrieve import retrieve_node
#联网搜索
from adaptive_search_rag.nodes.search import search_node
from adaptive_search_rag.nodes.generate import generate_node


def path_fn(state: RAGState) -> str:
    """条件路由：根据 state["route"] 决定下一步，非法值 fallback 到 retrieve"""
    route = state["route"]
    # TODO1：route=="search" 返回"search"，其余全部返回retrieve
    if route == "search":
        return "search"
    else:
        return "retrieve"


def build_graph(checkpointer=None, nodes=None):
    """组装图并 compile，返回可 invoke 的图"""
    # nodes 是可选注入，用于测试（默认 None → 用真实节点）
    n = nodes or {}          # 形如 {"route": fake, "retrieve": fake, ...}
    route = n.get("route", route_node)
    retrieve = n.get("retrieve", retrieve_node)
    search = n.get("search", search_node)
    generate = n.get("generate", generate_node)

    g = StateGraph(RAGState)

    # TODO2：add_node 注册四个节点
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("search", search)
    g.add_node("generate", generate)

    # TODO3：起始边：START → route节点
    g.add_edge(START, "route")

    # TODO4：条件分支：route节点执行完毕，走path_fn判断走向
    g.add_conditional_edges(
        "route",
        path_fn,
        {"retrieve": "retrieve", "search": "search"}
    )

    # TODO5：两条边，检索/搜索完成后都进入generate
    g.add_edge("retrieve", "generate")
    g.add_edge("search", "generate")

    # TODO6：生成答案后到END，图结束
    g.add_edge("generate", END)

    return g.compile(checkpointer=checkpointer)
