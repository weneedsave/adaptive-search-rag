from typing import get_type_hints
from adaptive_search_rag.state import RAGState


def test_state_fields_roundtrip():
    # TODO1 构造实例
    state = RAGState(
        question="你是谁？",
        route="retrieve",
        contexts=["我是吃白饭的大鲸鱼"],
        answer="我是哦鲸鲸",
    )
    # TODO2 断言读回
    assert state["question"] == "你是谁？"
    assert state["route"] == "retrieve"
    assert state["contexts"] == ["我是吃白饭的大鲸鱼"]
    assert state["answer"] == "我是哦鲸鲸"


def test_state_type_hints():
    # TODO3 反射获取类型提示
    hints = get_type_hints(RAGState)
    # TODO4 校验类型声明
    assert hints["question"] is str
    assert hints["route"] is str
    assert hints["contexts"] == list[str]
    assert hints["answer"] is str
