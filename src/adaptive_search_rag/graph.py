from langgraph.graph import StateGraph, START, END
# state状态定义
from adaptive_search_rag.state import RAGState
# 本地知识库检索
from adaptive_search_rag.nodes.retrieve import retrieve_node
# 检索结果评估节点
from adaptive_search_rag.nodes.evaluate import evaluate_node
# 查询改写节点
from adaptive_search_rag.nodes.rewrite import rewrite_query_node
# 联网搜索
from adaptive_search_rag.nodes.search import search_node
# LLM生成最终答案
from adaptive_search_rag.nodes.generate import generate_node

# 模块常量，放在最前面
MAX_RETRY = 1

def should_act(state: RAGState) -> str:
    grade = state.get("grade")
    valid_grades = ("correct", "ambiguous", "incorrect")
    if grade not in valid_grades:
        grade = "incorrect"

    if grade == "correct":
        return "generate"
    if grade == "ambiguous":
        return "search"

    # 只有grade=incorrect才执行下面，才读取retry_count
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRY:
        return "rewrite"
    else:
        return "search"


def build_graph(checkpointer=None, nodes=None):
    n = nodes or {}
    # 移除 route，只保留5个节点
    # if "retrieve" in n:
    #     retrieve = n["retrieve"]
    # else:
    #     retrieve = retrieve_node

    retrieve = n.get("retrieve", retrieve_node)
    evaluate = n.get("evaluate", evaluate_node)
    rewrite = n.get("rewrite", rewrite_query_node)
    search = n.get("search", search_node)
    generate = n.get("generate", generate_node)

    g = StateGraph(RAGState)
    # 注册节点
    g.add_node("retrieve", retrieve)
    g.add_node("evaluate", evaluate)
    g.add_node("rewrite", rewrite)
    g.add_node("search", search)
    g.add_node("generate", generate)

    # 边
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        should_act,
        path_map={
            "generate": "generate",
            "search": "search",
            "rewrite": "rewrite"
        }
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("search", "generate")
    g.add_edge("generate", END)

    if checkpointer:
        g = g.compile(checkpointer=checkpointer)
    else:
        g = g.compile()
    return g
