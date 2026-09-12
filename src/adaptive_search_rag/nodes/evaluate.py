from typing import Literal, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from adaptive_search_rag.state import RAGState
from adaptive_search_rag.llm import get_llm

class GradeDecision(TypedDict):
    grade: Literal["correct", "ambiguous", "incorrect"]

SYSTEM_PROMPT = """
你是RAG资料评估器。根据用户原始问题与参考资料，三选一判定：
1. correct：现有参考资料信息充足，足够回答用户问题
2. ambiguous：资料有相关内容，但信息不足，不足以完整回答；应补充联网搜索
3. incorrect：本地资料完全无关，无法回答；应先改写查询词重试本地检索
"""

def evaluate_node(state: RAGState, llm=None) -> dict:
    """
    评估当前contexts是否足够回答question
    返回字典，更新state的grade；incorrect/空context时清空contexts
    """
    question = state["question"]
    contexts = state["contexts"]

    if not contexts:
        #没有参考资料，判定为 incorrect
        return {"grade": "incorrect", "contexts": []}

    # LLM懒加载，仅在未注入时初始化（适配测试fake llm）
    if llm is None:
        llm = get_llm().with_structured_output(GradeDecision, method="function_calling")

    # 拼接参考资料，沿用generate的格式，带编号分隔
    context_text = "\n\n".join([f"【参考资料{i+1}】{chunk}" for i, chunk in enumerate(contexts)])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"用户问题：{question}\n参考资料：\n{context_text}")
    ]
    decision: GradeDecision = llm.invoke(messages)# 拿到结构化结果结构为 GradeDecision
    grade = decision["grade"]

    if grade == "incorrect":
        #交给 rewrite_node 重试本地检索
        return {"grade": grade, "contexts": []}
    else:
        # correct / ambiguous，context保持原样，只写入grade
        return {"grade": grade}
