# state.py
from typing import TypedDict, Literal

class RAGState(TypedDict):
    question: str           # 原始用户问题，全程固定不变；generate、web_search都读取这个
    retrieval_query: str    # 送入检索器的查询文本，rewrite节点更新；初始为空字符串
    contexts: list[str]
    grade: Literal["correct", "ambiguous", "incorrect"]  # 评估节点写入评估结果
    retry_count: int        # 本地检索改写重试次数，初始=0
    answer: str
