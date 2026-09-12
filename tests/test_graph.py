"""图级测试：CRAG 三档分流 + 改写重试环。
核心断言方式是**节点调用序列**（trace），而不是最终 answer：
answer 只能证明「跑完了」，trace 能证明「走的是哪条路」。
"""
from typing import Any
from adaptive_search_rag.graph import build_graph, should_act, MAX_RETRY
from adaptive_search_rag.nodes.retrieve import retrieve_node
from adaptive_search_rag.nodes.search import search_node

ORIGINAL = "RAG它解决了大模型什么问题？"
REWRITTEN = "RAG 大模型 问题"

# ============ 初始 state ============
def _state(**overrides: Any):
    """完整六字段的初始 state，overrides 覆盖个别字段。
    base = 六个字段的初值字典，照抄 main.py 里 app.invoke 的入参
    调用方只写想变的字段：_state(question="随便什么")
    """
    base = {
        "question": "",
        "retrieval_query": "",
        "contexts": [],
        "grade": "",
        "retry_count": 0,
        "answer": ""
    }
    return {**base, **overrides}

# ============ 第一块：should_act 单元测试 ============
# 纯函数，不碰图、不调 LLM，最便宜的一层。每个分支一条断言。
def test_should_act_correct():
    # grade="correct" → "generate"
    s = _state(grade="correct")
    assert should_act(s) == "generate"

def test_should_act_ambiguous():
    # grade="ambiguous" → "search"
    s = _state(grade="ambiguous")
    assert should_act(s) == "search"

def test_should_act_incorrect_retries_first():
    # grade="incorrect" 且 retry_count=0 → "rewrite"
    s = _state(grade="incorrect", retry_count=0)
    assert should_act(s) == "rewrite"

def test_should_act_incorrect_caps_out():
    # grade="incorrect" 且 retry_count=MAX_RETRY → "search"
    # 用 MAX_RETRY 而不是写死 1：锁的是「到达上限就转联网」这个不变量，
    # 而不是「上限等于 1」这个具体数值。
    s = _state(grade="incorrect", retry_count=MAX_RETRY)
    assert should_act(s) == "search"

def test_should_act_missing_grade_falls_back():
    # state 里没有 grade 键 → 按 incorrect 兜底（retry未满时 "rewrite"）
    s = {"retry_count": 0}
    assert should_act(s) == "rewrite"

def test_should_act_illegal_grade_falls_back():
    # grade="retrieval" 这种非法值 → 同样兜底为 incorrect
    s = _state(grade="retrieval", retry_count=0)
    assert should_act(s) == "rewrite"

def test_should_act_tolerates_missing_retry_count():
    """回归测试：曾经写成 state["retry_count"]，缺键时抛 KeyError。
    这一条的意义不是「现在对」，而是「以后不许再错」——
    谁把 .get 改回下标，这里立刻红。
    只传 {"grade": "correct"}，断言返回 "generate" 且不抛异常。
    注意：correct 分支压根不该读 retry_count——所以这条同时验证了
    「读取位置收窄到 incorrect 分支内」这个结构。
    """
    s = {"grade": "correct"}
    assert should_act(s) == "generate"
    assert should_act({"grade": "incorrect"}) == "rewrite"

# ============ 第二块：图级路径测试 ============
def _run(verdicts: list[str]):
    """跑一遍图，返回 (trace, queries, result)。
    verdicts: 第 N 次调用 evaluate 时该判什么，如 ["incorrect", "correct"]；
             超出长度后重复使用最后一个，方便表达「永远 incorrect」。
    trace:    节点名按调用顺序组成的 list
    queries:  retriever 实际收到的 query，按调用顺序
    """
    trace: list[str] = []
    queries: list[str] = []
    verdict_idx = 0

    def fake_retriever(query, top_k=3):
        nonlocal queries
        queries.append(query)
        return [f"chunk({query})"]

    def do_retrieve(state):
        trace.append("retrieve")
        return retrieve_node(state, retriever=fake_retriever)

    def fake_evaluate(state):
        nonlocal verdict_idx
        trace.append("evaluate")
        v = verdicts[min(verdict_idx, len(verdicts)-1)]
        verdict_idx += 1
        if v == "incorrect":
            # 按真实节点行为：incorrect 清空 contexts
            return {"grade": v, "contexts": []}
        else:
            return {"grade": v}

    def fake_rewrite(state):
        trace.append("rewrite")
        new_rc = state.get("retry_count", 0) + 1
        return {"retrieval_query": REWRITTEN, "retry_count": new_rc}

    def do_search(state):
        trace.append("search")
        fake_search_fn = lambda q, k=3: ["web结果"]
        return search_node(state, top_k=3, search_fn=fake_search_fn)

    def fake_generate(state):
        trace.append("generate")
        return {"answer": "ok"}

    nodes = {
        "retrieve": do_retrieve,
        "evaluate": fake_evaluate,
        "rewrite": fake_rewrite,
        "search": do_search,
        "generate": fake_generate,
    }
    graph = build_graph(nodes=nodes)
    result = graph.invoke(_state(question=ORIGINAL))
    return trace, queries, result


def test_path_correct_goes_straight_to_generate():
    # _run(["correct"]) 的 trace 应该是 ["retrieve", "evaluate", "generate"]
    trace, _, _ = _run(["correct"])
    assert trace == ["retrieve", "evaluate", "generate"]

def test_path_ambiguous_goes_to_search():
    # _run(["ambiguous"]) 的 trace 应该是 ["retrieve", "evaluate", "search", "generate"]
    trace, _, _ = _run(["ambiguous"])
    assert trace == ["retrieve", "evaluate", "search", "generate"]

def test_path_incorrect_loops_once_then_generates():
    # _run(["incorrect", "correct"])
    # trace: ["retrieve", "evaluate", "rewrite", "retrieve", "evaluate", "generate"]
    trace, _, _ = _run(["incorrect", "correct"])
    assert trace == ["retrieve", "evaluate", "rewrite", "retrieve", "evaluate", "generate"]

def test_path_incorrect_caps_out_to_search():
    # _run(["incorrect"]) 永远返回incorrect，达到MAX_RETRY上限走search
    trace, _, _ = _run(["incorrect"])
    expected = ["retrieve", "evaluate", "rewrite", "retrieve", "evaluate", "search", "generate"]
    assert trace == expected

def test_rewrite_runs_exactly_max_retry_times():
    # 环的封顶不变量：无论 evaluate 判多少次 incorrect，改写次数恰好等于上限。
    trace, _, _ = _run(["incorrect"])
    assert trace.count("rewrite") == MAX_RETRY

def test_loop_actually_changes_retrieval_query():
    """环的核心不变量：第二轮 retrieve 收到的 query 必须与第一轮不同。
    没有这一条，环可能是「空转」的——跑了两轮但送进去的是同一个 query。
    这个 bug 不会报错，只会静默地浪费一次检索 + 一次 LLM 调用。
    """
    _, queries, _ = _run(["incorrect", "correct"])
    assert len(queries) == 2
    assert queries[0] == ORIGINAL
    assert queries[1] == REWRITTEN
    assert queries[0] != queries[1]

def test_generate_still_sees_original_question():
    """改写只能影响检索，绝不许污染 question。"""
    _, _, result = _run(["incorrect", "correct"])
    assert result["question"] == ORIGINAL

# ============ 第三块：融合行为（真 search_node + 假 search_fn）============
def test_ambiguous_merges_local_and_web_contexts():
    """ambiguous 路径下，联网结果必须【追加】在本地结果之后。
    用真 search_node，所以测的是 search.py 里真实的 existing + new 拼接逻辑.
    """
    def fake_search_fn(q, top_k=3):
        return ["web结果"]
    state = _state(contexts=["本地片段"])
    out = search_node(state, search_fn=fake_search_fn)
    assert out["contexts"] == ["本地片段", "web结果"]

def test_search_failure_keeps_local_contexts():
    """联网失败时不能丢掉本地结果。
    返回 [] 会让 ambiguous 路径白检索一场，退化成「根据现有资料无法回答」。
    """
    def broken_search_fn(q, top_k=3):
        raise RuntimeError("web api down")
    state = _state(contexts=["本地片段"])
    out = search_node(state, search_fn=broken_search_fn)
    assert out["contexts"] == ["本地片段"]
