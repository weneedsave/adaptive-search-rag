from langchain_core.messages import SystemMessage, HumanMessage
from adaptive_search_rag.state import RAGState
from adaptive_search_rag.llm import get_llm

SYSTEM_PROMPT = """
你是查询改写专家。将用户问题改写为适合向量知识库检索的名词关键词短语。
规则：
1. 仅输出改写后的短语，不要解释、不要多余文字，不要引号。
2. 保留全部核心实体、专业名词；删掉口语化修饰、代词。
3. 严禁增加原文不存在的词语，不新增概念，不编造内容。

示例：
原始问题：RAG它解决了大模型什么问题？
改写：RAG 大模型 解决的问题

"""

def rewrite_query_node(state: RAGState, llm=None) -> dict:
    """把原问题改写成关键词式检索 query。
    返回恒定为两个键：{"retrieval_query": str, "retry_count": int}
    """
    if llm is None:
        llm = get_llm()

    question = state["question"]
    # 重试计数+1，无论LLM成功失败都增加次数，防止无限循环
    retry_count = state.get("retry_count", 0) + 1

    # TODO1: 组装消息
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"原始用户问题：{question}")
    ]
    try:
        # TODO2: 调用LLM，拿到改写后的query
        resp = llm.invoke(messages)
        new_query = resp.content.strip()#获得原始文本后去掉空格
    except Exception as e:
        print(f"【改写查询异常】{e}")
        new_query = ""

    # TODO3: 兜底，如果返回空，直接使用原始问题
    if not new_query:
        new_query = question
    #返回处理后的问题,重试次数
    return {"retrieval_query": new_query, "retry_count": retry_count}
