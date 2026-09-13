from __future__ import annotations
from functools import partial
from adaptive_search_rag.graph import build_graph
from adaptive_search_rag.nodes.search import search_node

#模拟联网产生的结果
STUB_MARKER = "[评测桩] 联网结果已省略"


def stub_search_fn(query: str, top_k: int = 3) -> list[str]:
    """替身。返回**单元素**列表，元素里同时含
    STUB_MARKER 和 query。
    格式：f"{STUB_MARKER}（query={query}）"
    为什么要把 query 写进去：search_node 用的是
    state["question"] 而非
    retrieval_query。把 query
    记进返回值，这条设计决策在评测链路里也被验证了。
    """
    text = f"{STUB_MARKER}（query={query}）"
    return [text]


def contains_stub(contexts: list[str]) -> bool:
    """任一元素包含 STUB_MARKER 即返回 True；空列表返回 False。
    这是判断「本题走了 search 分支」的依据 —— 比
    parse trace 简单可靠。
    """
    for doc in contexts:
        if STUB_MARKER in doc:
            return True
    return False


def build_eval_graph():
    """一行。
    build_graph(nodes={"search":
    partial(search_node, search_fn=stub_search_fn)})
    不传 checkpointer：评测不写
    checkpoints.sqlite，也避免跨题状态污染。
    """
    #调用作图,把联网函数替换成我们的虚假函数
    return build_graph(nodes={"search": partial(search_node, search_fn=stub_search_fn)})
