from typing import get_type_hints, Literal
from adaptive_search_rag.state import RAGState

def test_state_fields_roundtrip():
    # 状态快照：已经完成一轮查询改写后的状态，逻辑自洽
    state = RAGState(
        question="你是谁？",
        retrieval_query="RAG 大模型 问题",
        contexts=["我是吃白饭的大鲸鱼"],
        grade="ambiguous",
        retry_count=1,
        answer="我是哦鲸鲸",
    )
    # 断言读取所有字段
    assert state["question"] == "你是谁？"
    assert state["retrieval_query"] == "RAG 大模型 问题"
    assert state["contexts"] == ["我是吃白饭的大鲸鱼"]
    assert state["grade"] == "ambiguous"
    assert state["retry_count"] == 1
    assert state["answer"] == "我是哦鲸鲸"


def test_state_type_hints():
    # 反射读取RAGState类型注解
    hints = get_type_hints(RAGState)
    assert hints["question"] is str
    assert hints["retrieval_query"] is str
    assert hints["contexts"] == list[str]
    assert hints["grade"]==Literal["correct","ambiguous","incorrect"]
    assert hints["retry_count"] is int
    assert hints["answer"] is str
