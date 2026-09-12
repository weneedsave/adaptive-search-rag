# scripts/verify_graph.py
"""端到端验证 CRAG 图：真实 LLM + 真实混合检索 + 真实联网。
不注入任何假节点——这是上真 API 前的最后一关。
用 g.stream() 观察【实际执行路径】，再用终态字段交叉验证判定结果。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.graph import build_graph

# (说明, 问题) —— 三条用例对应三种期望路径
CASES = [
    ("库里有、问法直接", "什么是RAG？"),
    ("库里有、问法口语化", "RAG它解决了大模型什么问题？"),
    ("库里没有、需联网", "今天北京天气怎么样？"),
]


def _state(q):
    """六字段初始 state，照抄 main.py 的 invoke 入参。
    返回 dict（question 用参数 q，其余五个字段给初值）
    """
    return {
        "question": q,
        "retrieval_query": "",
        "contexts": [],
        "grade": "",
        "retry_count": 0,
        "answer": ""
    }


def run_case(g, question):
    """跑一个用例，返回 (path, final)。
    path:  节点名按执行顺序组成的 list（retrieve 出现两次 = 触发了环）
    final: 各节点 partial update 合并后的结果
    陷阱：final 是【累积的 updates】，不是完整 state。
    没有任何节点会写 question，所以 final 里【没有 "question" 这个键】。
    要打印问题直接用参数 question，别去 final["question"] 取。
    （也不要用 g.invoke() 再跑一遍拿完整 state——那会让每个用例
    多调一轮 LLM + 多跑一次精排，白花钱和时间。）
    """
    path = []
    final = {}
    for step in g.stream(_state(question)):
        # step 形如 {"节点名": {该节点返回的 partial state}}
        node_name, partial_state = next(iter(step.items()))
        path.append(node_name)
        final.update(partial_state)
    return path, final


def main():
    g = build_graph()   # 真实图：真 LLM + 真检索 + 真联网

    for label, q in CASES:
        print(f"【{label}】{q}")
        t0 = time.time()
        path, final = run_case(g, q)
        elapsed = time.time() - t0

        # 打印各项指标
        path_str = " -> ".join(path)
        grade = final.get("grade", "N/A")
        retry_count = final.get("retry_count", 0)
        retrieval_query = final.get("retrieval_query", "(未改写)")
        answer = final.get("answer", "")[:200]

        print(f"执行路径 : {path_str}")
        print(f"判定结果 : {grade}")
        print(f"重试次数 : {retry_count}")
        print(f"检索词   : {retrieval_query}")
        print(f"耗时     : {elapsed:.1f}s")
        print(f"答案     : {answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
