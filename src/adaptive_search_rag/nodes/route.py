from typing import Literal, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from adaptive_search_rag.state import RAGState
from adaptive_search_rag.llm import get_llm


class RouteDecision(TypedDict):
    route: Literal["retrieve", "search"]


SYSTEM_PROMPT = """
你是知识库问答系统的路由判定器。判断用户的问题应该走
哪条路径：
- "retrieve"：问题能在【本地知识库】里回答。本地知识
库包含这些主题：
  * RAG 原理（检索增强生成、幻觉、向量化、分块）
  * LangGraph 概念（StateGraph、节点、边、条件路由）
  * 智能体记忆（checkpoint、store、长期记忆）
- "search"：问题需要【联网搜索】才能回答。包括：
  * 实时/最新信息（今天、最近、股价、天气、新闻）
  * 本地知识库覆盖不到的特定事实查询
规则：
1. 只输出 retrieve 或 search，不要解释；
2. 如果问题涉及本地知识库里的主题 → retrieve；
3. 如果问题涉及实时信息或库外知识 → search；
4. 拿不准时优先 search（联网兜底更安全）。
"""


def route_node(state: RAGState, llm=None) -> dict:
    """判断问题属于知识库还是联网，返回 {"route": "retrieve" | "search"}"""
    if llm is None:
        llm = get_llm().with_structured_output(RouteDecision, method="function_calling")
    question = state["question"]
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    # 调用 llm.invoke(messages)，拿到结构化结果
    decision = llm.invoke(messages)
    #从 decision 里取 route 字段，返回 {"route": route}
    route = decision["route"]
    return {"route": route}
